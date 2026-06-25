"""
Training loop для DreamRec baseline.

Запуск:
    python src/train_dreamrec.py --config configs/dreamrec.yaml
"""

from __future__ import annotations

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from data import build_dataloaders
from evaluate import evaluate
from models.dreamrec import DreamRec


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class DreamRecConfig:
    # Данные
    data_dir:    str = './data/ATG'
    padding_idx: int = 11924
    n_items:     int = 11924
    batch_size:  int = 400
    num_workers: int = 4
    max_seq_len: int = 10

    # Модель
    d_model:       int   = 64
    n_heads:       int   = 1
    d_ff:          int   = 256
    dropout:       float = 0.1
    T:             int   = 200
    beta_schedule: str   = 'exp'
    p_uncond:         float = 0.1
    w_cfg:            float = 2.0
    retrieval_metric: str   = 'l2'   # 'l2', 'dot', 'cosine'

    # Оптимизация
    lr:            float = 5e-4
    weight_decay:  float = 0.0
    grad_clip:     float = 1.0
    num_epochs:    int   = 1000
    warmup_epochs: int   = 0

    # Воспроизводимость
    seed: int = 42

    # EMA
    ema_decay: float = 0.9999

    # Eval и сохранение
    eval_every_epochs: int   = 10
    top_k_values:      tuple = (10, 20)
    output_dir:        str   = './checkpoints'
    run_name:          str   = 'dreamrec_baseline'


# --------------------------------------------------------------------------- #
# EMA
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
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

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

def save_checkpoint(path, model, ema, optimizer, scheduler, epoch, step, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch, 'step': step,
        'model_state': model.state_dict(),
        'ema_shadow': ema.shadow,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'metrics': metrics,
        'model_cfg': {
            'n_items':          model.n_items,
            'd_model':          model.d_model,
            'T':                model.T,
            'w_cfg':            model.w_cfg,
            'p_uncond':         model.p_uncond,
            'retrieval_metric': model.retrieval_metric,
        },
    }, path)


def load_checkpoint(path, model, ema, optimizer, scheduler, device):
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

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def train(cfg: DreamRecConfig) -> None:
    set_seed(cfg.seed)
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
    print(f'  Steps per epoch: {steps_per_epoch}, Total: {total_steps:,}')

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print('Building DreamRec...')
    model = DreamRec(
        n_items=cfg.n_items,
        padding_idx=cfg.padding_idx,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
        T=cfg.T,
        beta_schedule=cfg.beta_schedule,
        p_uncond=cfg.p_uncond,
        w_cfg=cfg.w_cfg,
        retrieval_metric=cfg.retrieval_metric,
    ).to(device)

    summary = model.param_summary()
    print(f'  Trainable params: {summary["total_trainable"]:,}')

    # ------------------------------------------------------------------ #
    # EMA, Optimizer, Scheduler
    # ------------------------------------------------------------------ #
    ema = EMA(model, decay=cfg.ema_decay)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=get_lr_lambda(warmup_steps),
    )

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #
    start_epoch = 0
    start_step  = 0
    best_hr20   = 0.0
    latest_ckpt = out_dir / 'latest.pt'
    if latest_ckpt.exists():
        start_epoch, start_step = load_checkpoint(
            latest_ckpt, model, ema, optimizer, scheduler, device,
        )

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
            nn.utils.clip_grad_norm_(
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
            f'loss {avg_loss:.4f} | lr {lr_cur:.2e} | {elapsed:.0f}s'
        )
        train_logger.log({
            'epoch': epoch + 1, 'step': step,
            'loss': round(avg_loss, 6), 'lr': round(lr_cur, 8),
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

            torch.manual_seed(42)
            metrics = evaluate(
                model=model,
                dataloader=val_loader,
                top_k_values=cfg.top_k_values,
                num_sample_steps=cfg.T,   # передаём для совместимости, модель игнорирует
                w_cfg=cfg.w_cfg,
                device=device,
                show_progress=True,
            )

            ema.restore(model)
            model.train()

            print(f'Val loss: {val_loss:.4f} | {metrics}')
            val_logger.log({
                'epoch': epoch + 1, 'step': step,
                'val_loss': round(val_loss, 6),
                **metrics,
            })

            if metrics.get('HR@20', 0) > best_hr20:
                best_hr20 = metrics['HR@20']
                save_checkpoint(
                    out_dir / 'best.pt', model, ema, optimizer, scheduler,
                    epoch=epoch + 1, step=step, metrics=metrics,
                )
                print(f'New best HR@20: {best_hr20:.4f} — saved best.pt')

            save_checkpoint(
                out_dir / 'latest.pt', model, ema, optimizer, scheduler,
                epoch=epoch + 1, step=step, metrics=metrics,
            )
            print()

    # ------------------------------------------------------------------ #
    # Final eval на test
    # ------------------------------------------------------------------ #
    print('\n--- Final evaluation on test set ---')
    loaders_test = build_dataloaders(
        data_dir=cfg.data_dir, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, seq_len=cfg.max_seq_len, splits=('test',),
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
        num_sample_steps=cfg.T,
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
    default_cfg = Path(__file__).resolve().parent.parent / 'configs' / 'dreamrec.yaml'
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

    cfg = DreamRecConfig(
        data_dir    = d.get('data_dir',    DreamRecConfig.data_dir),
        padding_idx = d.get('padding_idx', DreamRecConfig.padding_idx),
        n_items     = d.get('n_items',     DreamRecConfig.n_items),
        batch_size  = d.get('batch_size',  DreamRecConfig.batch_size),
        num_workers = d.get('num_workers', DreamRecConfig.num_workers),
        max_seq_len = d.get('max_seq_len', DreamRecConfig.max_seq_len),
        d_model       = m.get('d_model',       DreamRecConfig.d_model),
        n_heads       = m.get('n_heads',       DreamRecConfig.n_heads),
        d_ff          = m.get('d_ff',          DreamRecConfig.d_ff),
        dropout       = float(m.get('dropout',       DreamRecConfig.dropout)),
        T             = int(m.get('T',               DreamRecConfig.T)),
        beta_schedule = str(m.get('beta_schedule',   DreamRecConfig.beta_schedule)),
        p_uncond      = float(m.get('p_uncond',      DreamRecConfig.p_uncond)),
        w_cfg             = float(m.get('w_cfg',             DreamRecConfig.w_cfg)),
        retrieval_metric  = str(m.get('retrieval_metric',   DreamRecConfig.retrieval_metric)),
        lr           = float(opt.get('lr',            DreamRecConfig.lr)),
        weight_decay = float(opt.get('weight_decay',  DreamRecConfig.weight_decay)),
        grad_clip    = (float(opt['grad_clip']) if opt.get('grad_clip') is not None else None),
        num_epochs    = int(opt.get('num_epochs',    DreamRecConfig.num_epochs)),
        warmup_epochs = int(opt.get('warmup_epochs', DreamRecConfig.warmup_epochs)),
        ema_decay     = float(ema.get('decay', DreamRecConfig.ema_decay)),
        seed          = int(raw.get('seed', DreamRecConfig.seed)),
        eval_every_epochs = int(ev.get('every_epochs', DreamRecConfig.eval_every_epochs)),
        top_k_values      = tuple(ev.get('top_k', list(DreamRecConfig.top_k_values))),
        output_dir = out.get('dir',      DreamRecConfig.output_dir),
        run_name   = out.get('run_name', DreamRecConfig.run_name),
    )

    train(cfg)
