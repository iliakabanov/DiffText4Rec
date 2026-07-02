"""
Training loop для DreamRecFMText (архитектура DreamRecText + Flow Matching вместо DDPM).

use_pca: true  → PCA-проекция 3072d → n_components (быстро, рекомендуется)
use_pca: false → сырые TEM 3072d (медленнее, для экспериментов)

Запуск:
    python src/train_dreamrec_fm_text.py --config configs/dreamrec_fm_text.yaml
"""

from __future__ import annotations

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import build_dataloaders
from evaluate import evaluate
from models.dreamrec_fm_text import DreamRecFMText


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class DreamRecFMTextConfig:
    # Данные
    data_dir:        str = './data/ATG'
    emb_path:        str = './data/ATG/3large_emb.pickle'
    whitening_cache: str = './data/processed/whitening_cache.npz'
    padding_idx:     int = 11924
    n_items:         int = 11924
    batch_size:      int = 256
    num_workers:     int = 4
    max_seq_len:     int = 10

    # Модель
    use_pca:                  bool  = True
    n_components:             int   = 256    # используется если use_pca=True
    per_component_whitening:  bool  = False  # True = Var(E)=I точно; False = глобальный std
    n_heads:          int   = 4
    d_ff:             int   = 1024
    dropout:          float = 0.1
    # FiLM denoiser
    denoiser_d_block:      int   = 256
    denoiser_d_ff:         int   = 1024
    denoiser_n_blocks:     int   = 3
    denoiser_time_emb_dim: int   = 128
    denoiser_dropout:      float = 0.1
    p_uncond:         float = 0.1
    w_cfg:            float = 2.0
    retrieval_metric: str   = 'l2'

    # Оптимизация
    lr:            float = 1e-3
    weight_decay:  float = 0.0
    grad_clip:     float = 1.0
    num_epochs:    int   = 300
    warmup_epochs: int   = 0

    # Воспроизводимость
    seed: int = 42

    # EMA
    ema_decay: float = 0.9999

    # Eval и сохранение
    eval_every_epochs: int   = 10
    top_k_values:      tuple = (10, 20)
    num_sample_steps:  int   = 50    # число шагов Euler ODE при сэмплинге
    output_dir:        str   = './checkpoints'
    run_name:          str   = 'dreamrec_fm_text'


# --------------------------------------------------------------------------- #
# Helpers (копия из train_dreamrec_text.py)
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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


def get_lr_lambda(warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return lr_lambda


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
            'n_items':                   model.n_items,
            'd_emb':                     model.d_emb,
            'use_pca':                   model.use_pca,
            'per_component_whitening':   model.per_component_whitening,
            'w_cfg':                     model.w_cfg,
            'p_uncond':                  model.p_uncond,
            'retrieval_metric':          model.retrieval_metric,
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

def train(cfg: DreamRecFMTextConfig) -> None:
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
    # Embeddings
    # ------------------------------------------------------------------ #
    print('Loading item embeddings...')
    with open(cfg.emb_path, 'rb') as f:
        emb_raw = pickle.load(f)
    if not isinstance(emb_raw, np.ndarray):
        raise ValueError(f'Expected numpy array, got {type(emb_raw)}')
    emb_raw = emb_raw.astype(np.float32)
    print(f'  Embeddings: {emb_raw.shape}, dtype={emb_raw.dtype}')

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print(f'Building DreamRecFMText (use_pca={cfg.use_pca})...')
    model = DreamRecFMText(
        item_embeddings_raw=emb_raw,
        padding_idx=cfg.padding_idx,
        use_pca=cfg.use_pca,
        n_components=cfg.n_components,
        per_component_whitening=cfg.per_component_whitening,
        whitening_cache_path=cfg.whitening_cache,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
        denoiser_d_block=cfg.denoiser_d_block,
        denoiser_d_ff=cfg.denoiser_d_ff,
        denoiser_n_blocks=cfg.denoiser_n_blocks,
        denoiser_time_emb_dim=cfg.denoiser_time_emb_dim,
        denoiser_dropout=cfg.denoiser_dropout,
        p_uncond=cfg.p_uncond,
        w_cfg=cfg.w_cfg,
        retrieval_metric=cfg.retrieval_metric,
    ).to(device)

    summary = model.param_summary()
    print(f'  d_emb={model.d_emb}')
    print(f'  Trainable params: {summary["total_trainable"]:,}')
    print(f'  Frozen params:    {summary["total_frozen"]:,}')

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
    start_epoch  = 0
    start_step   = 0
    best_ndcg20  = 0.0
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
                model=model, dataloader=val_loader,
                top_k_values=cfg.top_k_values,
                num_sample_steps=cfg.num_sample_steps,
                w_cfg=cfg.w_cfg,
                device=device, show_progress=True,
            )

            ema.restore(model)
            model.train()

            print(f'Val loss: {val_loss:.4f} | {metrics}')
            val_logger.log({
                'epoch': epoch + 1, 'step': step,
                'val_loss': round(val_loss, 6),
                **metrics,
            })

            if metrics.get('NDCG@20', 0) > best_ndcg20:
                best_ndcg20 = metrics['NDCG@20']
                save_checkpoint(
                    out_dir / 'best.pt', model, ema, optimizer, scheduler,
                    epoch=epoch + 1, step=step, metrics=metrics,
                )
                print(f'New best NDCG@20: {best_ndcg20:.4f} — saved best.pt')

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
        model=model, dataloader=loaders_test['test'],
        top_k_values=cfg.top_k_values,
        num_sample_steps=cfg.num_sample_steps,
        w_cfg=cfg.w_cfg,
        device=device, show_progress=True,
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
    default_cfg = Path(__file__).resolve().parent.parent / 'configs' / 'dreamrec_fm_text.yaml'
    parser.add_argument('--config', type=str, default=str(default_cfg))
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    d   = raw.get('data', {})
    m   = raw.get('model', {})
    den = raw.get('denoiser', {})
    opt = raw.get('optimizer', {})
    ema = raw.get('ema', {})
    ev  = raw.get('eval', {})
    out = raw.get('output', {})

    cfg = DreamRecFMTextConfig(
        data_dir        = d.get('data_dir',        DreamRecFMTextConfig.data_dir),
        emb_path        = d.get('emb_path',         DreamRecFMTextConfig.emb_path),
        whitening_cache = d.get('whitening_cache',  DreamRecFMTextConfig.whitening_cache),
        padding_idx     = d.get('padding_idx',      DreamRecFMTextConfig.padding_idx),
        n_items         = d.get('n_items',          DreamRecFMTextConfig.n_items),
        batch_size      = d.get('batch_size',       DreamRecFMTextConfig.batch_size),
        num_workers     = d.get('num_workers',      DreamRecFMTextConfig.num_workers),
        max_seq_len     = d.get('max_seq_len',      DreamRecFMTextConfig.max_seq_len),
        use_pca                 = bool(m.get('use_pca',                   DreamRecFMTextConfig.use_pca)),
        n_components            = int(m.get('n_components',               DreamRecFMTextConfig.n_components)),
        per_component_whitening = bool(m.get('per_component_whitening',   DreamRecFMTextConfig.per_component_whitening)),
        n_heads         = m.get('n_heads',               DreamRecFMTextConfig.n_heads),
        d_ff            = m.get('d_ff',                  DreamRecFMTextConfig.d_ff),
        dropout         = float(m.get('dropout',         DreamRecFMTextConfig.dropout)),
        denoiser_d_block      = int(den.get('d_block',       DreamRecFMTextConfig.denoiser_d_block)),
        denoiser_d_ff         = int(den.get('d_ff',          DreamRecFMTextConfig.denoiser_d_ff)),
        denoiser_n_blocks     = int(den.get('n_blocks',      DreamRecFMTextConfig.denoiser_n_blocks)),
        denoiser_time_emb_dim = int(den.get('time_emb_dim',  DreamRecFMTextConfig.denoiser_time_emb_dim)),
        denoiser_dropout      = float(den.get('dropout',     DreamRecFMTextConfig.denoiser_dropout)),
        p_uncond        = float(m.get('p_uncond',        DreamRecFMTextConfig.p_uncond)),
        w_cfg           = float(m.get('w_cfg',           DreamRecFMTextConfig.w_cfg)),
        retrieval_metric = str(m.get('retrieval_metric', DreamRecFMTextConfig.retrieval_metric)),
        lr           = float(opt.get('lr',            DreamRecFMTextConfig.lr)),
        weight_decay = float(opt.get('weight_decay',  DreamRecFMTextConfig.weight_decay)),
        grad_clip    = (float(opt['grad_clip']) if opt.get('grad_clip') is not None else None),
        num_epochs    = int(opt.get('num_epochs',    DreamRecFMTextConfig.num_epochs)),
        warmup_epochs = int(opt.get('warmup_epochs', DreamRecFMTextConfig.warmup_epochs)),
        ema_decay     = float(ema.get('decay', DreamRecFMTextConfig.ema_decay)),
        seed          = int(raw.get('seed', DreamRecFMTextConfig.seed)),
        eval_every_epochs = int(ev.get('every_epochs', DreamRecFMTextConfig.eval_every_epochs)),
        top_k_values      = tuple(ev.get('top_k', list(DreamRecFMTextConfig.top_k_values))),
        num_sample_steps  = int(ev.get('num_sample_steps', DreamRecFMTextConfig.num_sample_steps)),
        output_dir = out.get('dir',      DreamRecFMTextConfig.output_dir),
        run_name   = out.get('run_name', DreamRecFMTextConfig.run_name),
    )

    train(cfg)
