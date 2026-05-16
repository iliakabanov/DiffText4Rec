import torch

class ATGDataset(torch.utils.data.Dataset):
    """
    Загружает один split (train/val/test) из ATG.
    Возвращает пары (history_indices, target_index, mask).
    history_indices — тензор int64 длины 10 (с padding token).
    target_index — скаляр int64.
    mask — float тензор длины 10, 1 для реальных позиций, 0 для padding.
    """
    
    def __init__(self, pickle_path, padding_token_id, max_seq_len=10):
        """Читает данные и сохраняет их как numpy-массивы для быстрого индекса."""
    
    def __len__(self) -> int:
        ...
    
    def __getitem__(self, idx) -> dict:
        """
        Возвращает dict:
            {
              'history': LongTensor[10],
              'target':  LongTensor[],
              'mask':    FloatTensor[10],
            }
        """


def build_dataloaders(
    train_path, val_path, test_path,
    padding_token_id,
    batch_size, num_workers=4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Удобная фабрика, возвращает три DataLoader'а."""


def collate_batch(batch: list[dict]) -> dict:
    """Стандартный default_collate подойдёт; объявляем для ясности."""