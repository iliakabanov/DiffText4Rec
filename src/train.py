"""
Обучение любой модели из builders.MODELS — какой именно, сказано в yaml (model_type):

    python src/training.py --config configs/dreamrec.yaml
    python src/training.py --config configs/dreamrec_text.yaml

От модели нужно:
    compute_loss(history, target, mask)               -> скаляр
    recommend(history, mask, top_k, num_steps, w_cfg) -> (B, top_k)   (вызывает evaluate())
    config()                                          -> dict параметров, пишется в чекпойнт

fit():
    - сид, данные (шаффл train — через отдельный генератор), модель, AdamW + линейный warmup;
    - обучение в bfloat16 autocast с gradient clipping;
    - каждые eval.every_epochs эпох: val loss, метрики, строка в log_val.csv,
      best.pt (по val NDCG@20) и latest.pt;
    - продолжение с latest.pt, если он есть;
    - финальный тест на best.pt.

Сэмплинг на валидации и тесте засевается EVAL_SEED внутри torch.random.fork_rng:
метрики воспроизводимы по одному чекпойнту (и совпадают с пересчётом в ноутбуках),
а поток случайности обучения — шаффл, шум, негативы — при этом не сбрасывается.

Чекпойнты:
    latest.pt — веса + состояние оптимизатора и scheduler (для продолжения обучения);
    best.pt   — только веса, метрики и model_cfg.
"""

from __future__ import annotations

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import csv
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from builders import load_model_config
from config import TrainConfig
from data import build_dataloaders
from evaluate import evaluate

EVAL_SEED = 42


# --------------------------------------------------------------------------- #
# Случайность
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """
    Сиды всех генераторов, влияющих на обучение (инициализация весов, dropout, шум,
    негативы). Шаффл train засевается отдельно — генератором в build_dataloaders.

    Побитовой воспроизводимости это не даёт (atomicAdd в backward nn.Embedding,
    bfloat16 autocast), но делает прогоны сравнимыми.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


@contextmanager
def seeded(seed: int, device: torch.device):
    """Засевает генераторы на время блока и восстанавливает их состояние после."""
    devices = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


# --------------------------------------------------------------------------- #
# Логи и чекпойнты
# --------------------------------------------------------------------------- #

class CSVLogger:
    """Дописывает строки в CSV; заголовок пишется только в новый файл."""

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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    step: int,
    metrics: dict[str, float],
    model_cfg: dict,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> None:
    """Без optimizer — только веса (best.pt); с ним — ещё и состояние для resume (latest.pt)."""
    state = {
        'epoch':       epoch,
        'step':        step,
        'model_state': model.state_dict(),
        'metrics':     metrics,
        'model_cfg':   model_cfg,
    }
    if optimizer is not None:
        state['optimizer_state'] = optimizer.state_dict()
        state['scheduler_state'] = scheduler.state_dict()
    torch.save(state, path)


def _resume(out_dir: Path, model: nn.Module, optimizer, scheduler,
            device: torch.device) -> tuple[int, int, float]:
    """Продолжение с latest.pt. Возвращает (эпоха, шаг, лучший val NDCG@20)."""
    latest = out_dir / 'latest.pt'
    if not latest.exists():
        return 0, 0, 0.0

    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])

    # Лучшую метрику берём из best.pt — иначе первая же валидация перезапишет best.pt
    best = out_dir / 'best.pt'
    best_ndcg20 = (torch.load(best, map_location='cpu', mmap=True)['metrics'].get('NDCG@20', 0.0)
                   if best.exists() else 0.0)
    print(f'Resumed from epoch {ckpt["epoch"]}, step {ckpt["step"]}, '
          f'best val NDCG@20 {best_ndcg20:.4f}')
    return ckpt['epoch'], ckpt['step'], best_ndcg20


# --------------------------------------------------------------------------- #
# Обучение
# --------------------------------------------------------------------------- #

def _warmup(warmup_steps: int) -> Callable[[int], float]:
    """Множитель lr: линейный рост за warmup_steps шагов, дальше константа."""
    def lr_lambda(step: int) -> float:
        return step / warmup_steps if step < warmup_steps else 1.0
    return lr_lambda


@torch.no_grad()
def _mean_loss(model: nn.Module, loader, device: torch.device, use_amp: bool) -> float:
    total = 0.0
    for batch in loader:
        history, mask, target = (batch[k].to(device) for k in ('history', 'mask', 'target'))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            total += model.compute_loss(history, target, mask).item()
    return total / len(loader)


def fit(
    cfg: TrainConfig,
    build_model: Callable[[TrainConfig], nn.Module],
) -> dict[str, float]:
    """
    Обучает модель по конфигу и возвращает метрики на test.

    Args:
        cfg:         конфиг (см. config.TrainConfig).
        build_model: cfg -> модель; вызывается после set_seed и создания лоадеров.
    """
    if 20 not in cfg.eval.top_k:
        raise ValueError(f'eval.top_k={cfg.eval.top_k} должен содержать 20: best.pt выбирается по NDCG@20')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, seed: {cfg.seed}')
    set_seed(cfg.seed)   # до создания модели и лоадеров

    out_dir = Path(cfg.output.dir) / cfg.output.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_logger = CSVLogger(out_dir / 'log_train.csv')
    val_logger   = CSVLogger(out_dir / 'log_val.csv')

    # ------------------------------------------------------------------ #
    # Данные
    # ------------------------------------------------------------------ #
    d = cfg.data
    loaders = build_dataloaders(
        data_dir=d.data_dir, batch_size=d.batch_size, num_workers=d.num_workers,
        seq_len=d.max_seq_len, seed=cfg.seed,
    )
    train_loader, val_loader = loaders['train'], loaders['val']

    opt = cfg.optimizer
    steps_per_epoch = len(train_loader)
    print(f'Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,} | '
          f'Test: {len(loaders["test"].dataset):,} sequences | '
          f'{steps_per_epoch} steps/epoch × {opt.num_epochs} epochs')

    # ------------------------------------------------------------------ #
    # Модель и оптимизатор
    # ------------------------------------------------------------------ #
    model  = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'{type(model).__name__}: trainable {sum(p.numel() for p in params):,}, frozen {n_frozen:,}')
    model_cfg = model.config()      # в чекпойнт, его читают ноутбуки анализа

    optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_warmup(opt.warmup_epochs * steps_per_epoch),
    )

    start_epoch, step, best_ndcg20 = _resume(out_dir, model, optimizer, scheduler, device)
    if start_epoch >= opt.num_epochs:
        print(f'Обучение уже завершено ({start_epoch}/{opt.num_epochs} эпох) — только финальный тест')

    use_amp      = device.type == 'cuda'
    grad_clip    = opt.grad_clip if opt.grad_clip is not None else float('inf')
    sample_steps = cfg.eval.num_sample_steps or 1       # SASRec не сэмплирует
    w_cfg        = getattr(cfg.model, 'w_cfg', 0.0)

    def run_eval(loader) -> dict[str, float]:
        with seeded(EVAL_SEED, device):
            return evaluate(model=model, dataloader=loader, top_k_values=cfg.eval.top_k,
                            num_sample_steps=sample_steps, w_cfg=w_cfg, device=device,
                            show_progress=True)

    # ------------------------------------------------------------------ #
    # Обучение
    # ------------------------------------------------------------------ #
    t_start = time.time()
    for epoch in range(start_epoch, opt.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            history, mask, target = (batch[k].to(device) for k in ('history', 'mask', 'target'))
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = model.compute_loss(history, target, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        elapsed  = time.time() - t_start
        lr       = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1:4d}/{opt.num_epochs} | loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s')
        train_logger.log({'epoch': epoch + 1, 'step': step, 'loss': round(avg_loss, 6),
                          'lr': round(lr, 8), 'elapsed_s': round(elapsed)})

        if (epoch + 1) % cfg.eval.every_epochs != 0:
            continue

        # ── Валидация ──────────────────────────────────────────────────
        print(f'\n--- Eval after epoch {epoch + 1} ---')
        model.eval()
        val_loss = _mean_loss(model, val_loader, device, use_amp)
        metrics  = run_eval(val_loader)
        model.train()

        print(f'Val loss: {val_loss:.4f} | {metrics}')
        val_logger.log({'epoch': epoch + 1, 'step': step, 'val_loss': round(val_loss, 6),
                        **metrics})

        ckpt = dict(model=model, epoch=epoch + 1, step=step, metrics=metrics,
                    model_cfg=model_cfg)
        if metrics['NDCG@20'] > best_ndcg20:
            best_ndcg20 = metrics['NDCG@20']
            save_checkpoint(out_dir / 'best.pt', **ckpt)
            print(f'New best NDCG@20: {best_ndcg20:.4f} — saved best.pt')
        save_checkpoint(out_dir / 'latest.pt', **ckpt, optimizer=optimizer, scheduler=scheduler)
        print()

    # ------------------------------------------------------------------ #
    # Финальный тест на best.pt
    # ------------------------------------------------------------------ #
    print('\n--- Final evaluation on test set (best.pt) ---')
    best_path = out_dir / 'best.pt'
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location='cpu', mmap=True)['model_state'])
    model.eval()
    test_metrics = run_eval(loaders['test'])

    print(f'Test: {test_metrics}')
    # Пустой val_loss: строка test совпадает по столбцам с заголовком
    val_logger.log({'epoch': 'test', 'step': step, 'val_loss': '', **test_metrics})
    print('\nDone.')
    return test_metrics


def main() -> dict[str, float]:
    """Точка входа: по model_type из yaml берёт модель из builders.MODELS и обучает её."""
    parser = argparse.ArgumentParser(description='Обучение модели по yaml-конфигу.')
    parser.add_argument('--config', required=True, help='configs/<модель>.yaml')
    args = parser.parse_args()

    spec, cfg = load_model_config(args.config)
    return fit(cfg, spec.build)


if __name__ == '__main__':
    main()
