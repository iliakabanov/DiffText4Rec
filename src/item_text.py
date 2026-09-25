"""
Текстовое описание товара Amazon Reviews 2023 для текстовых эмбеддингов (TEM).

Шаблон (пустые поля пропускаются целиком):

    title: <title>
    store_title: <store>
    features: <пункты features через cfg.feature_sep>
    categories: <путь категорий через cfg.category_sep>

Два варианта текста:
    text_raw — поля как есть, только схлопнуты пробельные символы;
    text     — после чистки по правилам из TextConfig: обрезка features, эмодзи и ™ ® ©,
               служебные префиксы пунктов, (опционально) капслок-заголовки пунктов.
               Чистка ничего не дописывает, только убирает шум; title, store и categories
               не обрезаются.

Общий для scripts/preprocess.py (items.parquet) и notebooks/toys_raw_structure.ipynb (раздел 5.5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

META_COLUMNS = ['parent_asin', 'title', 'store', 'features', 'categories']


@dataclass(frozen=True)
class TextConfig:
    field_sep:          str  = '\n'     # между полями
    feature_sep:        str  = ' | '    # между пунктами features (';' встречается внутри пунктов)
    category_sep:       str  = ' > '    # между уровнями пути категорий
    max_features:       int  = 5        # пунктов features, не больше
    max_features_chars: int  = 800      # символов во всех пунктах features вместе
    strip_symbols:      bool = True     # эмодзи и ™ ® ©
    strip_prefixes:     bool = True     # «Features:», «Product Description -» в начале пункта
    fix_caps_headers:   bool = False    # «CHILD SAFETY - text» → «Child safety: text» (портит бренды)


SYMBOLS_RE = re.compile('[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍™®©]')
PREFIX_RE  = re.compile(r'^(?:about this item|product description|description|key features|features?)'
                        r'\s*[:\-–—]\s*', re.IGNORECASE)
# Заголовок пункта: 3–60 символов без строчных латинских букв, затем «-», «:» или тире и текст
CAPS_HEADER_RE = re.compile(r'^([^a-z:]{3,60}?)\s*[:\-–—]+\s+(?=\S)')


# --------------------------------------------------------------------------- #
# Поля
# --------------------------------------------------------------------------- #

def clean(v) -> str:
    """Строка без лишних пробельных символов; не-строки (None, NaN) → ''."""
    return ' '.join(v.split()) if isinstance(v, str) else ''


def clean_list(v) -> list[str]:
    """Список строк (list или np.ndarray из parquet) без пустых; остальное → []."""
    return [c for c in map(clean, v) if c] if isinstance(v, (list, np.ndarray)) else []


def clean_str(v, cfg: TextConfig) -> str:
    s = clean(v)
    return clean(SYMBOLS_RE.sub(' ', s)) if cfg.strip_symbols else s


def clean_feature(v, cfg: TextConfig) -> str:
    s = clean_str(v, cfg)
    if cfg.strip_prefixes:
        s = PREFIX_RE.sub('', s)
    if cfg.fix_caps_headers:
        head = CAPS_HEADER_RE.match(s)
        if head and sum(ch.isalpha() for ch in head.group(1)) >= 2:
            s = f'{head.group(1).strip().capitalize()}: {s[head.end():]}'
    return s


def clean_features(v, cfg: TextConfig) -> list[str]:
    """Первые пункты целиком, пока не превышены max_features и max_features_chars; без повторов.
    Если длиннее лимита уже первый пункт, он обрезается по границе слова."""
    out, seen, total = [], set(), 0
    for s in (clean_feature(x, cfg) for x in clean_list(v)):
        if not s or s.lower() in seen:
            continue
        if len(out) == cfg.max_features or (out and total + len(s) > cfg.max_features_chars):
            break
        if len(s) > cfg.max_features_chars:
            s = s[:cfg.max_features_chars].rsplit(' ', 1)[0] + '...'
        out.append(s)
        seen.add(s.lower())
        total += len(s)
    return out


# --------------------------------------------------------------------------- #
# Текст товара
# --------------------------------------------------------------------------- #

def format_text(title: str, store: str, features: list[str], categories: list[str],
                cfg: TextConfig) -> str:
    fields = [
        ('title',       title),
        ('store_title', store),
        ('features',    cfg.feature_sep.join(features)),
        ('categories',  cfg.category_sep.join(categories)),
    ]
    return cfg.field_sep.join(f'{name}: {value}' for name, value in fields if value)


def raw_text(title, store, features, categories, cfg: TextConfig) -> str:
    return format_text(clean(title), clean(store), clean_list(features), clean_list(categories), cfg)


def cleaned_text(title, store, features, categories, cfg: TextConfig) -> str:
    return format_text(clean_str(title, cfg), clean_str(store, cfg), clean_features(features, cfg),
                       [c for c in (clean_str(x, cfg) for x in clean_list(categories)) if c], cfg)


def popularity(parent_asin: pd.Series) -> pd.Series:
    """Число взаимодействий товара по убыванию; при равенстве — по parent_asin (порядок детерминирован)."""
    counts = parent_asin.astype(str).value_counts().sort_index()
    return counts.sort_values(ascending=False, kind='stable').rename_axis('parent_asin')


def build_item_texts(meta: pd.DataFrame, cfg: TextConfig = TextConfig()) -> pd.DataFrame:
    """
    Метаданные с колонками META_COLUMNS → DataFrame (parent_asin, text, text_raw) в том же порядке.
    Товары без метаданных (NaN в полях) получают пустой текст.
    """
    rows = list(zip(meta['title'], meta['store'], meta['features'], meta['categories']))
    return pd.DataFrame({
        'parent_asin': meta['parent_asin'].to_numpy(),
        'text':        [cleaned_text(*r, cfg) for r in rows],
        'text_raw':    [raw_text(*r, cfg) for r in rows],
    })


# --------------------------------------------------------------------------- #
# Quick sanity check:  cd src && python item_text.py
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    cfg = TextConfig()
    assert clean_str('LEGO® City set™ 🏎', cfg) == 'LEGO City set'
    assert clean_feature('Features: Discover & Play Piano', cfg) == 'Discover & Play Piano'
    assert clean_feature('CHILD SAFETY - Meet US Safety Standards.', cfg) == 'CHILD SAFETY - Meet US Safety Standards.'
    caps = TextConfig(fix_caps_headers=True)
    assert clean_feature('CHILD SAFETY - Meet US Safety Standards.', caps) == 'Child safety: Meet US Safety Standards.'
    assert clean_feature('Colorful Baby Toys: 8 cups', caps) == 'Colorful Baby Toys: 8 cups'

    feats = np.array(['a ' * 300, 'b', 'b', 'c'], dtype=object)       # как из parquet
    short = TextConfig(max_features_chars=100)
    out = clean_features(feats, short)
    assert len(out) == 1 and out[0].endswith('...') and len(out[0]) <= 103
    assert clean_features(['x', 'x', 'y'], cfg) == ['x', 'y']
    assert len(clean_features([f'f{k}' for k in range(10)], cfg)) == cfg.max_features

    meta = pd.DataFrame({'parent_asin': ['A', 'B'], 'title': ['Toy  car', None], 'store': ['Brand', None],
                         'features': [['fast', 'red'], None], 'categories': [['Toys', 'Cars'], None]})
    texts = build_item_texts(meta, cfg)
    assert texts.loc[0, 'text'] == 'title: Toy car\nstore_title: Brand\nfeatures: fast | red\ncategories: Toys > Cars'
    assert texts.loc[1, 'text'] == texts.loc[1, 'text_raw'] == ''
    print('All checks passed')
