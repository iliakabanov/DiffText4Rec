"""
Обучение любой модели из builders.MODELS — какой именно, сказано в yaml (model_type):

    python src/training.py --config configs/dreamrec.yaml
    python src/training.py --config configs/dreamrec_text.yaml

От модели нужно:
    compute_loss(history, target, mask)               -> скаляр
    recommend(history, mask, top_k, num_steps, w_cfg) -> (B, top_k)   (вызывает evaluate())
    config()                                          -> dict параметров, пишется в чекпойнт

fit():
    - сид, данные (шаффл train — через отдельный генератор), модель, Adam или AdamW
      (optimizer.name) + линейный warmup;
    - обучение в bfloat16 autocast с gradient clipping;
    - при optimizer.ema_decay — EMA обучаемых весов (класс EMA): валидация, best.pt и тест
      идут по усреднённым весам, а метрики сырых весов пишутся в log_val.csv как raw_*;
    - каждые eval.every_epochs эпох: val loss, метрики, строка в log_val.csv,
      best.pt (по val NDCG@20) и latest.pt;
    - продолжение с latest.pt, если он есть;
    - финальный тест на best.pt.

Сэмплинг на валидации и тесте засевается EVAL_SEED внутри torch.random.fork_rng:
метрики воспроизводимы по одному чекпойнту (и совпадают с пересчётом в ноутбуках),
а поток случайности обучения — шаффл, шум, негативы — при этом не сбрасывается.

Чекпойнты:
    latest.pt — сырые веса + состояние оптимизатора, scheduler и EMA (для продолжения обучения);
    best.pt   — только веса (при EMA — усреднённые), метрики и model_cfg.
"""

from __future__ import annotations

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import csv
import random
import time
from contextlib import contextmanager, nullcontext
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

# weight_decay у них означает разное:
#   adam  — L2: wd·θ прибавляется к градиенту и нормируется Adam вместе с ним. У редко
#           обновляемых строк эмбеддингов √v мало, и затухание выходит порядка lr за шаг.
#   adamw — развязанное затухание θ ← θ − lr·wd·θ, одинаковое для всех параметров.
OPTIMIZERS = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}


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
# EMA весов
# --------------------------------------------------------------------------- #

class EMA:
    """
    Экспоненциальное скользящее среднее обучаемых весов — теневые веса θ̄:

        θ̄ ← β_t·θ̄ + (1 − β_t)·θ  после каждого шага оптимизатора,
        β_t = min(decay, (1 + t) / (10 + t)).

    Растущее β_t даёт короткое окно в начале, чтобы в θ̄ не застревала случайная инициализация;
    дальше полураспад вклада старых весов — ln 2 / (1 − decay) шагов. Обучение идёт по θ,
    оценка — по θ̄ внутри applied(). Усредняются только параметры с requires_grad:
    замороженные эмбеддинги и буферы не меняются, их копировать незачем.
    """

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f'ema_decay должен быть в (0, 1), получено {decay}')
        self.decay  = decay
        self.step   = 0
        named       = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.names  = [n for n, _ in named]
        self.params = [p for _, p in named]
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self) -> None:
        self.step += 1
        beta = min(self.decay, (1 + self.step) / (10 + self.step))
        for s, p in zip(self.shadow, self.params):
            s.lerp_(p.detach(), 1.0 - beta)

    @contextmanager
    def applied(self):
        """На время блока параметры модели — теневые веса; после блока сырые веса возвращаются."""
        self._swap()
        try:
            yield
        finally:
            self._swap()

    def _swap(self) -> None:
        # Обмен тензорами вместо копирования: без второй копии весов в памяти.
        # Параметры остаются теми же объектами, так что состояние оптимизатора не затрагивается
        for i, p in enumerate(self.params):
            p.data, self.shadow[i] = self.shadow[i], p.data

    def state_dict(self) -> dict:
        return {'decay': self.decay, 'step': self.step, 'shadow': dict(zip(self.names, self.shadow))}

    def load_state_dict(self, state: dict) -> None:
        if set(state['shadow']) != set(self.names):
            raise ValueError('EMA в чекпойнте не совпадает с обучаемыми параметрами модели')
        self.step = state['step']
        for s, name in zip(self.shadow, self.names):
            s.copy_(state['shadow'][name])


def _raw_keys(top_k: tuple[int, ...]) -> list[str]:
    """Столбцы log_val.csv с val loss и метриками сырых весов (при EMA)."""
    return ['raw_val_loss'] + [f'raw_{m}@{k}' for k in top_k for m in ('HR', 'NDCG')]


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
    ema: EMA | None = None,
) -> None:
    """
    Без optimizer — только веса (best.pt); с ним — ещё и состояние для resume (latest.pt).
    Сохраняются текущие параметры модели: внутри ema.applied() это усреднённые веса.
    """
    state = {
        'epoch':       epoch,
        'step':        step,
        'model_state': model.state_dict(),
        'metrics':     metrics,
        'model_cfg':   model_cfg,
        'ema_decay':   ema.decay if ema is not None else None,
    }
    if optimizer is not None:
        state['optimizer_state'] = optimizer.state_dict()
        state['scheduler_state'] = scheduler.state_dict()
        if ema is not None:
            state['ema_state'] = ema.state_dict()
    torch.save(state, path)


def _resume(out_dir: Path, model: nn.Module, optimizer, scheduler, ema: EMA | None,
            device: torch.device) -> tuple[int, int, float]:
    """Продолжение с latest.pt. Возвращает (эпоха, шаг, лучший val NDCG@20)."""
    latest = out_dir / 'latest.pt'
    if not latest.exists():
        return 0, 0, 0.0

    ckpt = torch.load(latest, map_location=device)
    # load_state_dict перезаписал бы флаг decoupled_weight_decay из чекпойнта, и Adam
    # молча работал бы как AdamW (или наоборот)
    saved_group = ckpt['optimizer_state']['param_groups'][0]
    current     = optimizer.param_groups[0].get('decoupled_weight_decay')
    if saved_group.get('decoupled_weight_decay', current) != current:
        raise ValueError(f'{latest}: оптимизатор в чекпойнте не совпадает с optimizer.name в конфиге; '
                         f'смените output.run_name или верните прежний оптимизатор')
    # best.pt и метрики прогона считались либо по сырым, либо по усреднённым весам — не смешиваем
    if (ema is not None) != ('ema_state' in ckpt):
        raise ValueError(f'{latest}: EMA {"есть" if "ema_state" in ckpt else "нет"} в чекпойнте, '
                         f'а optimizer.ema_decay={ema.decay if ema is not None else None} в конфиге; '
                         f'смените output.run_name или верните прежнюю настройку')
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    if ema is not None:
        if ckpt['ema_state']['decay'] != ema.decay:
            print(f'EMA decay меняется с {ckpt["ema_state"]["decay"]} на {ema.decay}')
        ema.load_state_dict(ckpt['ema_state'])

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

    if opt.name not in OPTIMIZERS:
        raise ValueError(f'optimizer.name должен быть одним из {list(OPTIMIZERS)}, получено {opt.name!r}')
    optimizer = OPTIMIZERS[opt.name](params, lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_warmup(opt.warmup_epochs * steps_per_epoch),
    )
    ema = EMA(model, opt.ema_decay) if opt.ema_decay is not None else None
    if ema is not None:
        print(f'EMA: decay {ema.decay}, полураспад {np.log(2) / (1 - ema.decay):.0f} шагов '
              f'(≈{np.log(2) / (1 - ema.decay) / steps_per_epoch:.1f} эпох)')
    ema_weights = ema.applied if ema is not None else nullcontext

    start_epoch, step, best_ndcg20 = _resume(out_dir, model, optimizer, scheduler, ema, device)
    if start_epoch >= opt.num_epochs:
        print(f'Обучение уже завершено ({start_epoch}/{opt.num_epochs} эпох) — только финальный тест')

    use_amp      = device.type == 'cuda'
    grad_clip    = opt.grad_clip if opt.grad_clip is not None else float('inf')
    sample_steps = cfg.eval.num_sample_steps or 1       # SASRec не сэмплирует
    w_cfg        = getattr(cfg.model, 'w_cfg', 0.0)

    def run_val_loss() -> float:
        # Засеян, как и сэмплинг: не расходует поток случайности обучения, поэтому траектория
        # не зависит от того, сколько раз считается val loss (с EMA — дважды)
        with seeded(EVAL_SEED, device):
            return _mean_loss(model, val_loader, device, use_amp)

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
            if ema is not None:
                ema.update()
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
        raw = {}
        if ema is not None:
            # Сырые веса — для контроля выигрыша/отставания EMA; сиды сэмплинга те же, сравнение парное
            raw_loss, raw_metrics = run_val_loss(), run_eval(val_loader)
            raw = dict(zip(_raw_keys(cfg.eval.top_k), [round(raw_loss, 6), *raw_metrics.values()]))
            print(f'Raw weights: val loss {raw_loss:.4f} | {raw_metrics}')

        ckpt = dict(model=model, epoch=epoch + 1, step=step, model_cfg=model_cfg, ema=ema)
        with ema_weights():
            val_loss = run_val_loss()
            metrics  = run_eval(val_loader)
            if metrics['NDCG@20'] > best_ndcg20:
                best_ndcg20 = metrics['NDCG@20']
                save_checkpoint(out_dir / 'best.pt', metrics=metrics, **ckpt)
                print(f'New best NDCG@20: {best_ndcg20:.4f} — saved best.pt')
        model.train()

        print(f'Val loss: {val_loss:.4f} | {metrics}' + (' (EMA)' if ema is not None else ''))
        val_logger.log({'epoch': epoch + 1, 'step': step, 'val_loss': round(val_loss, 6),
                        **metrics, **raw})
        save_checkpoint(out_dir / 'latest.pt', metrics=metrics, **ckpt,
                        optimizer=optimizer, scheduler=scheduler)
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
    # Пустые val_loss и raw_*: строка test совпадает по столбцам с заголовком
    raw_empty = dict.fromkeys(_raw_keys(cfg.eval.top_k), '') if ema is not None else {}
    val_logger.log({'epoch': 'test', 'step': step, 'val_loss': '', **test_metrics, **raw_empty})
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
