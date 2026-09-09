"""
Training loop для SASRec baseline.

Структура идентична train.py:
    - EMA весов
    - Mixed precision (bfloat16)
    - Linear warmup + constant lr
    - Gradient clipping
    - Валидация каждые eval_every_epochs эпох
    - Лучший чекпойнт по NDCG@20 на val
    - Логирование в CSV
    - Resume из latest.pt

Запуск:
    cd src && python train_sasrec.py --config ../configs/sasrec.yaml
"""

from __future__ import annotations

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import build_dataloaders
from evaluate import evaluate
from models.sasrec import SASRec


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class SASRecConfig:
    """Все гиперпараметры SASRec в одном месте."""

    # Данные
    data_dir:    str = './data/ATG'
    padding_idx: int = 11924
    n_items:     int = 11924
    batch_size:  int = 256
    num_workers: int = 4
    max_seq_len: int = 10

    # Модель
    d_model:     int   = 128
    n_heads:     int   = 4
    n_layers:    int   = 1
    d_ff:        int   = 512
    dropout:     float = 0.2
    tie_weights: bool  = True
    loss:        str   = 'infonce'
    neg_ratio:   int   = 64
    temperature: float = 0.07

    # Оптимизация
    lr:            float = 1e-3
    weight_decay:  float = 1e-6
    grad_clip:     float = 1.0
    num_epochs:    int   = 500
    warmup_epochs: int   = 2

    # EMA
    ema_decay: float = 0.9999

    # Воспроизводимость
    seed: int = 42

    # Eval и сохранение
    eval_every_epochs: int   = 10
    top_k_values:      tuple = (10, 20)
    output_dir:        str   = './checkpoints'
    run_name:          str   = 'sasrec_baseline'


# --------------------------------------------------------------------------- #
# Воспроизводимость
# --------------------------------------------------------------------------- #

def setup_seed(seed: int) -> None:
    """
    Фиксирует все источники случайности, которые влияют на обучение:
      - инициализацию весов (normal_/xavier_uniform_ в SASRec._init_weights),
      - маски dropout,
      - сэмплинг негативов в InfoNCE (torch.randint).

    Порядок шаффла задаётся отдельно — через generator в build_dataloaders.

    Побитовой воспроизводимости это НЕ даёт: backward у nn.Embedding
    использует atomicAdd, порядок суммирования градиентов по повторяющимся
    индексам на GPU не определён. Плюс автотюнинг cuBLAS и bfloat16 autocast.
    Сид делает прогоны сравнимыми, но не идентичными — метрики baseline
    всё равно нужно репортить как mean ± std по нескольким сидам.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# EMA (идентично train.py)
# --------------------------------------------------------------------------- #

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> None:
        self._backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}


# --------------------------------------------------------------------------- #
# LR Schedule
# --------------------------------------------------------------------------- #

def get_lr_lambda(warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return lr_lambda


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #

def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch':           epoch,
        'step':            step,
        'model_state':     model.state_dict(),
        'ema_shadow':      ema.shadow,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'metrics':         metrics,
    }, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    ema.shadow = {k: v.to(device) for k, v in ckpt['ema_shadow'].items()}
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    epoch = ckpt.get('epoch', 0)
    step  = ckpt.get('step', 0)
    print(f'Resumed from epoch {epoch}, step {step}, metrics: {ckpt["metrics"]}')
    return epoch, step


# --------------------------------------------------------------------------- #
# CSV Logger
# --------------------------------------------------------------------------- #

class CSVLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    def log(self, row: dict) -> None:
        write_header = not self._initialized and not self.path.exists()
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._initialized = True


# --------------------------------------------------------------------------- #
# Main train function
# --------------------------------------------------------------------------- #

def train(cfg: SASRecConfig) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # До создания модели и лоадеров — иначе инициализация весов уже уйдёт
    # на глобальном RNG, засеянном энтропией ОС.
    setup_seed(cfg.seed)
    print(f'Seed: {cfg.seed}')

    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_logger = CSVLogger(out_dir / 'log_train.csv')
    val_logger   = CSVLogger(out_dir / 'log_val.csv')

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    print('Loading data...')
    loaders = build_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seq_len=cfg.max_seq_len,
        seed=cfg.seed,
    )
    train_loader = loaders['train']
    val_loader   = loaders['val']

    steps_per_epoch = len(train_loader)
    warmup_steps    = cfg.warmup_epochs * steps_per_epoch
    total_steps     = cfg.num_epochs * steps_per_epoch

    print(f'  Train: {len(train_loader.dataset):,} sequences')
    print(f'  Val:   {len(val_loader.dataset):,} sequences')
    print(f'  Steps per epoch: {steps_per_epoch}')
    print(f'  Total steps:     {total_steps:,}  ({cfg.num_epochs} epochs)')
    print(f'  Warmup steps:    {warmup_steps}  ({cfg.warmup_epochs} epochs)')

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print('Building model...')
    model = SASRec(
        n_items=cfg.n_items,
        padding_idx=cfg.padding_idx,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
        tie_weights=cfg.tie_weights,
        loss=cfg.loss,
        neg_ratio=cfg.neg_ratio,
        temperature=cfg.temperature,
    ).to(device)

    summary = model.param_summary()
    print(f'  Trainable params: {summary["total_trainable"]:,}')

    # ------------------------------------------------------------------ #
    # EMA, Optimizer, Scheduler
    # ------------------------------------------------------------------ #
    ema = EMA(model, decay=cfg.ema_decay)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_lr_lambda(warmup_steps),
    )

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #
    start_epoch = 0
    start_step  = 0
    best_ndcg20 = 0.0
    latest_ckpt = out_dir / 'latest.pt'
    if latest_ckpt.exists():
        start_epoch, start_step = load_checkpoint(
            latest_ckpt, model, ema, optimizer, scheduler, device
        )

    # ------------------------------------------------------------------ #
    # Mixed precision
    # ------------------------------------------------------------------ #
    use_amp   = device.type == 'cuda'
    amp_dtype = torch.bfloat16

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    print(f'\nStarting training from epoch {start_epoch + 1}/{cfg.num_epochs}...')

    step    = start_step
    t_start = time.time()

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            history = batch['history'].to(device)
            mask    = batch['mask'].to(device)
            target  = batch['target'].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model.compute_loss(history, target, mask)

            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.grad_clip if cfg.grad_clip is not None else float('inf'),
            )

            optimizer.step()
            scheduler.step()
            ema.update(model)

            step += 1
            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        elapsed  = time.time() - t_start
        lr_cur   = scheduler.get_last_lr()[0]

        print(
            f'Epoch {epoch + 1:4d}/{cfg.num_epochs} | '
            f'loss {avg_loss:.4f} | '
            f'lr {lr_cur:.2e} | '
            f'{elapsed:.0f}s'
        )
        train_logger.log({
            'epoch':     epoch + 1,
            'step':      step,
            'loss':      round(avg_loss, 6),
            'lr':        round(lr_cur, 8),
            'elapsed_s': round(elapsed),
        })

        # ------------------------------------------------------------------ #
        # Evaluation
        # ------------------------------------------------------------------ #
        if (epoch + 1) % cfg.eval_every_epochs == 0:
            print(f'\n--- Eval after epoch {epoch + 1} ---')

            ema.apply(model)
            model.eval()

            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    history = batch['history'].to(device)
                    mask    = batch['mask'].to(device)
                    target  = batch['target'].to(device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        val_loss += model.compute_loss(history, target, mask).item()
            val_loss /= len(val_loader)

            # Никакого manual_seed здесь: recommend() у SASRec ничего не
            # сэмплирует, так что ресид только сбрасывал глобальный RNG в одно
            # и то же состояние каждые eval_every_epochs эпох — порядок шаффла
            # и поток негативов InfoNCE начинали повторяться одним циклом.
            metrics = evaluate(
                model=model,
                dataloader=val_loader,
                top_k_values=cfg.top_k_values,
                num_sample_steps=1,    # SASRec не использует
                w_cfg=0.0,             # SASRec не использует
                device=device,
                show_progress=True,
            )

            train_metrics = evaluate(
                model=model,
                dataloader=train_loader,
                top_k_values=(20,),
                num_sample_steps=1,
                w_cfg=0.0,
                device=device,
                show_progress=False,
            )

            ema.restore(model)
            model.train()

            train_ndcg20 = train_metrics['NDCG@20']
            print(f'Val loss: {val_loss:.4f} | {metrics} | train_NDCG@20: {train_ndcg20:.4f}')
            val_logger.log({
                'epoch':         epoch + 1,
                'step':          step,
                'val_loss':      round(val_loss, 6),
                'train_ndcg20':  round(train_ndcg20, 6),
                **metrics,
            })

            if metrics.get('NDCG@20', 0) > best_ndcg20:
                best_ndcg20 = metrics['NDCG@20']
                save_checkpoint(
                    out_dir / 'best.pt',
                    model, ema, optimizer, scheduler,
                    epoch=epoch + 1, step=step, metrics=metrics,
                )
                print(f'New best NDCG@20: {best_ndcg20:.4f} — saved best.pt')

            save_checkpoint(
                out_dir / 'latest.pt',
                model, ema, optimizer, scheduler,
                epoch=epoch + 1, step=step, metrics=metrics,
            )
            print()

    # ------------------------------------------------------------------ #
    # Final eval на test
    # ------------------------------------------------------------------ #
    print('\n--- Final evaluation on test set ---')
    loaders_test = build_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seq_len=cfg.max_seq_len,
        splits=('test',),
    )

    best_ckpt = out_dir / 'best.pt'
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, ema, optimizer, scheduler, device)

    ema.apply(model)
    model.eval()

    test_metrics = evaluate(
        model=model,
        dataloader=loaders_test['test'],
        top_k_values=cfg.top_k_values,
        num_sample_steps=1,
        w_cfg=0.0,
        device=device,
        show_progress=True,
    )
    ema.restore(model)

    print(f'Test: {test_metrics}')
    val_logger.log({'epoch': 'test', 'step': step, **test_metrics})
    print('\nDone.')


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    default_cfg = Path(__file__).resolve().parent.parent / 'configs' / 'sasrec.yaml'
    parser.add_argument('--config', type=str, default=str(default_cfg))
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    d   = raw.get('data', {})
    m   = raw.get('model', {})
    opt = raw.get('optimizer', {})
    ema = raw.get('ema', {})
    ev  = raw.get('eval', {})
    out = raw.get('output', {})

    cfg = SASRecConfig(
        data_dir    = d.get('data_dir',    SASRecConfig.data_dir),
        padding_idx = d.get('padding_idx', SASRecConfig.padding_idx),
        n_items     = d.get('n_items',     SASRecConfig.n_items),
        batch_size  = d.get('batch_size',  SASRecConfig.batch_size),
        num_workers = d.get('num_workers', SASRecConfig.num_workers),
        max_seq_len = d.get('max_seq_len', SASRecConfig.max_seq_len),
        d_model     = m.get('d_model',     SASRecConfig.d_model),
        n_heads     = m.get('n_heads',     SASRecConfig.n_heads),
        n_layers    = m.get('n_layers',    SASRecConfig.n_layers),
        d_ff        = m.get('d_ff',        SASRecConfig.d_ff),
        dropout     = float(m.get('dropout',     SASRecConfig.dropout)),
        tie_weights = bool(m.get('tie_weights',  SASRecConfig.tie_weights)),
        loss        = str(m.get('loss',          SASRecConfig.loss)),
        neg_ratio   = int(m.get('neg_ratio',     SASRecConfig.neg_ratio)),
        temperature = float(m.get('temperature', SASRecConfig.temperature)),
        lr           = float(opt.get('lr',            SASRecConfig.lr)),
        weight_decay = float(opt.get('weight_decay',  SASRecConfig.weight_decay)),
        grad_clip    = (float(opt['grad_clip']) if opt.get('grad_clip') is not None
                        else None),
        num_epochs    = int(opt.get('num_epochs',    SASRecConfig.num_epochs)),
        warmup_epochs = int(opt.get('warmup_epochs', SASRecConfig.warmup_epochs)),
        ema_decay     = float(ema.get('decay', SASRecConfig.ema_decay)),
        seed          = int(raw.get('seed', SASRecConfig.seed)),
        eval_every_epochs = int(ev.get('every_epochs', SASRecConfig.eval_every_epochs)),
        top_k_values      = tuple(ev.get('top_k', list(SASRecConfig.top_k_values))),
        output_dir = out.get('dir',      SASRecConfig.output_dir),
        run_name   = out.get('run_name', SASRecConfig.run_name),
    )

    train(cfg)
