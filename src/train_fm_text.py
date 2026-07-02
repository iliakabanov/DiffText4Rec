"""
Training loop для FlowMatcher.

Особенности:
    - Тренировка по эпохам (num_epochs).
    - EMA (Exponential Moving Average) весов.
    - Mixed precision (bfloat16).
    - Linear warmup по эпохам.
    - Gradient clipping.
    - Валидация каждые eval_every_epochs эпох с EMA-весами.
    - Сохранение лучшего чекпойнта по NDCG@20 на val.
    - Логирование в CSV.
    - Resume из чекпойнта.
"""

from __future__ import annotations

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# ROOT = Path(__file__).resolve().parent.parent
# sys.path.append(str(ROOT))

import csv
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import build_dataloaders
from evaluate import evaluate
from models.flow_matcher import FlowMatcher


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    """Все гиперпараметры в одном месте."""

    # Данные
    data_dir: str = './data/ATG'
    emb_path: str = './data/ATG/3large_emb.pickle'
    whitening_cache: str = './data/processed/whitening_cache.npz'
    padding_idx: int = 11924
    n_components: int = 256       # PCA размерность для FM
    batch_size: int = 256
    num_workers: int = 4
    max_seq_len: int = 10

    # Encoder
    d_hidden: int = 512
    d_cond: int = 512
    encoder_layers: int = 2
    encoder_heads: int = 8
    encoder_d_ff: int = 2048
    encoder_dropout: float = 0.1

    # Denoiser
    denoiser_d_block: int = 512        # меньше чем при 3072d — задача проще
    denoiser_d_ff: int = 1024
    denoiser_n_blocks: int = 6
    denoiser_time_emb_dim: int = 128
    denoiser_dropout: float = 0.0

    # Flow Matching
    p_uncond:                float = 0.1
    retrieval_metric:        str   = 'dot'   # 'dot', 'cosine' или 'l2'
    per_component_whitening: bool  = False   # True = Var(E)=I точно; False = глобальный std

    # Оптимизация
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_epochs: int = 200          # основной параметр — число эпох
    warmup_epochs: int = 2         # число эпох линейного warmup

    # EMA
    ema_decay: float = 0.9999

    # Воспроизводимость
    seed: int = 42

    # Eval и сохранение
    eval_every_epochs: int = 5     # оцениваем каждые N эпох
    top_k_values: tuple = (10, 20)
    num_sample_steps: int = 50
    w_cfg: float = 1.0
    output_dir: str = './checkpoints'
    run_name: str = 'baseline_2'


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #

class EMA:
    """
    Exponential Moving Average параметров модели.

    Хранит отдельную копию весов (shadow), обновляет после каждого шага:
        shadow = decay * shadow + (1 - decay) * param

    Для inference используем EMA-веса — они более сглаженные и стабильные.

    Usage:
        ema = EMA(model, decay=0.9999)
        # После optimizer.step():
        ema.update(model)
        # Для eval:
        ema.apply(model)      # копируем EMA-веса в модель
        ... eval ...
        ema.restore(model)    # возвращаем обучаемые веса обратно
    """

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
        """Обновляет EMA-веса после optimizer.step()."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: nn.Module) -> None:
        """Копирует EMA-веса в модель для eval/inference."""
        self._backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Восстанавливает обучаемые веса из бэкапа после eval."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}


# --------------------------------------------------------------------------- #
# LR Schedule
# --------------------------------------------------------------------------- #

def get_lr_lambda(warmup_steps: int) -> callable:
    """
    Linear warmup по шагам, затем constant lr.

    Args:
        warmup_steps: число шагов линейного нагрева
                      (вычисляется как warmup_epochs * steps_per_epoch).
    """
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
    """Сохраняет чекпойнт."""
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
    """
    Загружает чекпойнт.

    Returns:
        (epoch, step) — с которых продолжать.
    """
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
    """Простой CSV-логгер для train loss и val метрик."""

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

def train(cfg: TrainConfig) -> None:
    """
    Главная функция тренировки (по эпохам).
    """
    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

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
    # Embeddings
    # ------------------------------------------------------------------ #
    print('Loading item embeddings...')
    import pickle
    with open(cfg.emb_path, 'rb') as f:
        emb_raw = pickle.load(f)
    if not isinstance(emb_raw, np.ndarray):
        raise ValueError(f'Expected numpy array, got {type(emb_raw)}')
    emb_raw = emb_raw.astype(np.float32)
    print(f'  Embeddings: {emb_raw.shape}, dtype={emb_raw.dtype}')

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print('Building model...')
    model = FlowMatcher(
        item_embeddings_raw=emb_raw,
        padding_idx=cfg.padding_idx,
        n_components=cfg.n_components,
        per_component_whitening=cfg.per_component_whitening,
        whitening_cache_path=cfg.whitening_cache,
        d_hidden=cfg.d_hidden,
        d_cond=cfg.d_cond,
        encoder_layers=cfg.encoder_layers,
        encoder_heads=cfg.encoder_heads,
        encoder_d_ff=cfg.encoder_d_ff,
        encoder_dropout=cfg.encoder_dropout,
        max_seq_len=cfg.max_seq_len,
        denoiser_d_block=cfg.denoiser_d_block,
        denoiser_d_ff=cfg.denoiser_d_ff,
        denoiser_n_blocks=cfg.denoiser_n_blocks,
        denoiser_time_emb_dim=cfg.denoiser_time_emb_dim,
        denoiser_dropout=cfg.denoiser_dropout,
        p_uncond=cfg.p_uncond,
        retrieval_metric=cfg.retrieval_metric,
    ).to(device)

    summary = model.param_summary()
    print(f'  Trainable params: {summary["total_trainable"]:,}')
    print(f'  Frozen params:    {summary["total_frozen"]:,}')

    # ------------------------------------------------------------------ #
    # EMA, Optimizer, Scheduler
    # ------------------------------------------------------------------ #
    ema = EMA(model, decay=cfg.ema_decay)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Warmup считается в шагах (warmup_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_lr_lambda(warmup_steps),
    )

    # ------------------------------------------------------------------ #
    # Resume из чекпойнта если есть
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
    # Training loop (по эпохам)
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

        # Средний loss за эпоху
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
            'epoch':      epoch + 1,
            'step':       step,
            'loss':       round(avg_loss, 6),
            'lr':         round(lr_cur, 8),
            'elapsed_s':  round(elapsed),
        })

        # ------------------------------------------------------------------ #
        # Evaluation (каждые eval_every_epochs эпох)
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

            torch.manual_seed(42)
            metrics = evaluate(
                model=model,
                dataloader=val_loader,
                top_k_values=cfg.top_k_values,
                num_sample_steps=cfg.num_sample_steps,
                w_cfg=cfg.w_cfg,
                device=device,
                show_progress=True,
            )

            torch.manual_seed(42)
            train_metrics = evaluate(
                model=model,
                dataloader=train_loader,
                top_k_values=(20,),
                num_sample_steps=cfg.num_sample_steps,
                w_cfg=cfg.w_cfg,
                device=device,
                show_progress=False,
            )

            ema.restore(model)
            model.train()

            train_ndcg20 = train_metrics['NDCG@20']
            print(f'Val loss: {val_loss:.4f} | {metrics} | train_NDCG@20: {train_ndcg20:.4f}')
            val_logger.log({
                'epoch': epoch + 1, 'step': step,
                'val_loss': round(val_loss, 6),
                'train_ndcg20': round(train_ndcg20, 6),
                **metrics,
            })

            # Лучший чекпойнт
            if metrics.get('NDCG@20', 0) > best_ndcg20:
                best_ndcg20 = metrics['NDCG@20']
                save_checkpoint(
                    out_dir / 'best.pt',
                    model, ema, optimizer, scheduler,
                    epoch=epoch + 1, step=step, metrics=metrics,
                )
                print(f'New best NDCG@20: {best_ndcg20:.4f} — saved best.pt')

            # Последний чекпойнт для resume
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

    # Загружаем лучшие веса
    best_ckpt = out_dir / 'best.pt'
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, ema, optimizer, scheduler, device)

    ema.apply(model)
    model.eval()

    test_metrics = evaluate(
        model=model,
        dataloader=loaders_test['test'],
        top_k_values=cfg.top_k_values,
        num_sample_steps=cfg.num_sample_steps,
        w_cfg=cfg.w_cfg,
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
    default_cfg = Path(__file__).resolve().parent.parent / 'configs' / 'fm_text.yaml'
    parser.add_argument('--config', type=str, default=str(default_cfg))
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    d   = raw.get('data', {})
    enc = raw.get('encoder', {})
    den = raw.get('denoiser', {})
    fm  = raw.get('flow_matching', {})
    opt = raw.get('optimizer', {})
    ema = raw.get('ema', {})
    ev  = raw.get('eval', {})
    out = raw.get('output', {})
    rep = raw.get('reproducibility', {})

    cfg = TrainConfig(
        data_dir          = d.get('data_dir',        TrainConfig.data_dir),
        emb_path          = d.get('emb_path',         TrainConfig.emb_path),
        whitening_cache   = d.get('whitening_cache',  TrainConfig.whitening_cache),
        padding_idx       = d.get('padding_idx',      TrainConfig.padding_idx),
        n_components      = d.get('n_components',     TrainConfig.n_components),
        batch_size        = d.get('batch_size',       TrainConfig.batch_size),
        num_workers       = d.get('num_workers',      TrainConfig.num_workers),
        max_seq_len       = d.get('max_seq_len',      TrainConfig.max_seq_len),
        d_hidden          = enc.get('d_hidden',       TrainConfig.d_hidden),
        d_cond            = enc.get('d_cond',         TrainConfig.d_cond),
        encoder_layers    = enc.get('layers',         TrainConfig.encoder_layers),
        encoder_heads     = enc.get('heads',          TrainConfig.encoder_heads),
        encoder_d_ff      = enc.get('d_ff',           TrainConfig.encoder_d_ff),
        encoder_dropout   = enc.get('dropout',        TrainConfig.encoder_dropout),
        denoiser_d_block  = den.get('d_block',        TrainConfig.denoiser_d_block),
        denoiser_d_ff     = den.get('d_ff',           TrainConfig.denoiser_d_ff),
        denoiser_n_blocks = den.get('n_blocks',       TrainConfig.denoiser_n_blocks),
        denoiser_time_emb_dim = den.get('time_emb_dim', TrainConfig.denoiser_time_emb_dim),
        denoiser_dropout  = den.get('dropout',        TrainConfig.denoiser_dropout),
        p_uncond                = float(fm.get('p_uncond',                TrainConfig.p_uncond)),
        retrieval_metric        = str(fm.get('retrieval_metric',         TrainConfig.retrieval_metric)),
        per_component_whitening = bool(fm.get('per_component_whitening', TrainConfig.per_component_whitening)),
        w_cfg                   = float(fm.get('w_cfg',                  TrainConfig.w_cfg)),
        lr                = float(opt.get('lr',             TrainConfig.lr)),
        weight_decay      = float(opt.get('weight_decay',   TrainConfig.weight_decay)),
        grad_clip         = (float(opt['grad_clip']) if opt.get('grad_clip') is not None
                             else None),
        num_epochs        = int(opt.get('num_epochs',     TrainConfig.num_epochs)),
        warmup_epochs     = int(opt.get('warmup_epochs',  TrainConfig.warmup_epochs)),
        ema_decay         = float(ema.get('decay',          TrainConfig.ema_decay)),
        eval_every_epochs = int(ev.get('every_epochs',    TrainConfig.eval_every_epochs)),
        top_k_values      = tuple(ev.get('top_k',    list(TrainConfig.top_k_values))),
        num_sample_steps  = int(ev.get('num_sample_steps', TrainConfig.num_sample_steps)),
        output_dir        = out.get('dir',            TrainConfig.output_dir),
        run_name          = out.get('run_name',       TrainConfig.run_name),
        seed              = int(rep.get('seed',       TrainConfig.seed)),
    )

    train(cfg)