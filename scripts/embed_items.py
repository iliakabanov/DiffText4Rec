"""
Текстовые эмбеддинги товаров из items.parquet (результат scripts/preprocess.py).

Строка i матрицы — товар в строке i items.parquet: порядок строк и есть индекс товара.
Формат выхода как у data/ATG/3large_emb.pickle — pickle с np.ndarray (N, D) float32, так что
data.load_item_embeddings читает его без изменений.

Считается кусками по CHUNK_SIZE товаров; каждый кусок сохраняется в OUT_DIR/<имя>_chunks/,
и при повторном запуске готовые куски пропускаются — прерванный прогон продолжается с места
остановки. Рядом с чанками лежит info.json (модель, колонка, число товаров, хэш порядка
parent_asin): если items.parquet или параметры поменялись, продолжение запрещено.

    python scripts/embed_items.py                      # колонка TEXT_COLUMN
    python scripts/embed_items.py --column text_raw
    python scripts/embed_items.py --limit 2000         # проверка на первых 2000 товарах
"""
import argparse
import hashlib
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Параметры
# --------------------------------------------------------------------------- #

ITEMS_PATH  = ROOT / 'data/processed/toys/items.parquet'
OUT_DIR     = ROOT / 'data/processed/toys'

MODEL       = 'Qwen/Qwen3-Embedding-0.6B'
TEXT_COLUMN = 'text'            # text — после чистки; text_raw — как есть
PROMPT      = None              # инструкция перед текстом; None — без неё (товары кодируются как документы)
MAX_LENGTH  = 1024              # токенов; тексты длиннее обрезаются (в лог пишется, сколько таких)
BATCH_SIZE  = 32
CHUNK_SIZE  = 10_000            # товаров в одном сохраняемом куске
DTYPE       = torch.bfloat16    # веса модели на GPU; эмбеддинги сохраняются во float32
NORMALIZE   = True              # L2-нормировка эмбеддингов

log = logging.getLogger('embed_items')


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #

def out_name(column: str) -> str:
    """Имя выхода: <модель>_<колонка>, например qwen3-embedding-0.6b_text."""
    return f'{MODEL.split("/")[-1].lower()}_{column}'


def run_info(items: pd.DataFrame, column: str) -> dict:
    """То, что должно совпадать у всех кусков одного прогона."""
    order = hashlib.sha1('\n'.join(items['parent_asin']).encode()).hexdigest()
    return {'model': MODEL, 'column': column, 'prompt': PROMPT, 'max_length': MAX_LENGTH,
            'normalize': NORMALIZE, 'n_items': len(items), 'parent_asin_sha1': order}


def load_model() -> SentenceTransformer:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(
        MODEL, device=device,
        model_kwargs={'torch_dtype': DTYPE if device == 'cuda' else torch.float32,
                      'attn_implementation': 'sdpa'},
        processor_kwargs={'padding_side': 'left'},     # рекомендация авторов: пулинг по последнему токену
    )
    assert model.tokenizer.padding_side == 'left', model.tokenizer.padding_side
    model.max_seq_length = MAX_LENGTH
    log.info(f'{MODEL} на {device}: размерность {model.get_sentence_embedding_dimension()}, '
             f'max_seq_length {model.max_seq_length}')
    return model


def encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    kwargs = {'prompt': PROMPT} if PROMPT else {}
    emb = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=NORMALIZE,
                       convert_to_numpy=True, show_progress_bar=False, **kwargs)
    return emb.astype(np.float32)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description='Эмбеддинги текстов товаров из items.parquet.')
    parser.add_argument('--column', default=TEXT_COLUMN, choices=['text', 'text_raw'])
    parser.add_argument('--limit', type=int, default=None, help='только первые N товаров (проверка)')
    args = parser.parse_args()

    name = out_name(args.column) + (f'_first{args.limit}' if args.limit else '')
    chunk_dir = OUT_DIR / f'{name}_chunks'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(chunk_dir / 'embed.log', mode='a', encoding='utf-8')],
    )

    items = pd.read_parquet(ITEMS_PATH, columns=['parent_asin', args.column])
    if args.limit:
        items = items.iloc[:args.limit]
    texts = items[args.column].tolist()
    log.info(f'{ITEMS_PATH.relative_to(ROOT)}: {len(items):,} товаров, колонка {args.column}; '
             f'пустых текстов {sum(t == "" for t in texts):,}')

    info = run_info(items, args.column)
    info_path = chunk_dir / 'info.json'
    if info_path.exists() and json.loads(info_path.read_text()) != info:
        raise ValueError(f'{info_path}: куски посчитаны для других items.parquet или параметров; '
                         f'удалите {chunk_dir}, чтобы начать заново')
    info_path.write_text(json.dumps(info, indent=1))

    model = load_model()
    n_long = sum(len(ids) > MAX_LENGTH for ids in model.tokenizer(texts[:5000])['input_ids'])
    log.info(f'Длиннее {MAX_LENGTH} токенов среди первых {min(5000, len(texts)):,} текстов: {n_long}')

    starts = range(0, len(texts), CHUNK_SIZE)
    t0, n_done = time.time(), 0
    for k, start in enumerate(starts):
        path = chunk_dir / f'chunk_{k:05d}.npy'
        if path.exists():
            continue
        emb = encode(model, texts[start:start + CHUNK_SIZE])
        np.save(path, emb)
        n_done += len(emb)
        rate = n_done / (time.time() - t0)
        left = (len(texts) - start - len(emb)) / rate
        log.info(f'кусок {k + 1}/{len(starts)}: товары {start:,}–{start + len(emb):,} | '
                 f'{rate:.0f} товаров/с | осталось ~{left / 60:.0f} мин')

    emb = np.concatenate([np.load(chunk_dir / f'chunk_{k:05d}.npy') for k in range(len(starts))])
    assert emb.shape[0] == len(items), (emb.shape, len(items))
    out = OUT_DIR / f'{name}.pickle'
    with open(out, 'wb') as f:
        pickle.dump(emb, f)
    norms = np.linalg.norm(emb, axis=1)
    log.info(f'Сохранено: {out.relative_to(ROOT)} {emb.shape} {emb.dtype}; '
             f'норма строк {norms.min():.3f}–{norms.max():.3f}; NaN: {int(np.isnan(emb).sum())}')


if __name__ == '__main__':
    main()
