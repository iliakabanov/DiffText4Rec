"""
Эмбеддинги товаров из чекпойнта SASRec → pickle в формате текстовых эмбеддингов.

Выход — pickle с np.ndarray (N_items, d_model) float32, строка i — товар i (без padding), как у
scripts/embed_items.py. Его читает data.load_item_embeddings, поэтому модели на «текстовых»
эмбеддингах (dreamrec_text, dreamrec_fm_text) берут его через data.emb_path без изменений кода:
PCA/whitening, перемешивание строк и оценка работают как с текстом.

Поддерживается SASRec с собственной таблицей item_embedding (не SASRecAlphaFuse).

    python scripts/export_sasrec_embeddings.py --checkpoint checkpoints/toys/sasrec_d256/best.pt
    python scripts/export_sasrec_embeddings.py --checkpoint ... --out data/processed/toys/my_emb.pickle
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description='Таблица эмбеддингов товаров SASRec → pickle.')
    parser.add_argument('--checkpoint', required=True, help='best.pt / latest.pt обученного SASRec')
    parser.add_argument('--out', help='куда сохранить; по умолчанию data/processed/toys/<прогон>_item_emb.pickle')
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['model_cfg']
    if cfg.get('model') != 'SASRec':
        raise ValueError(f'{ckpt_path}: ожидался SASRec с собственной таблицей, получен {cfg.get("model")!r}')

    n_items = cfg['n_items']
    weight  = ckpt['model_state']['item_embedding.weight']          # (n_items + 1, d), последняя — padding
    emb = weight[:n_items].float().numpy().astype(np.float32)

    out = Path(args.out) if args.out else ROOT / f'data/processed/toys/{ckpt_path.parent.name}_item_emb.pickle'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(emb, f)

    norms = np.linalg.norm(emb, axis=1)
    print(f'{ckpt_path} (эпоха {ckpt["epoch"]}, val {ckpt["metrics"]})')
    print(f'Эмбеддинги {emb.shape}, норма: mean {norms.mean():.3f}, min {norms.min():.3f}, '
          f'max {norms.max():.3f}; loss_metric={cfg.get("loss_metric")}, '
          f'retrieval_metric={cfg.get("retrieval_metric")}')
    print(f'Сохранено: {out}')


if __name__ == '__main__':
    main()
