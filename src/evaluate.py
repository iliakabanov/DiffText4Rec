@torch.no_grad()
def evaluate(
    flow_matcher: FlowMatcher,
    dataloader: DataLoader,
    top_k_values: tuple = (10, 20),
    num_sample_steps: int = 50,
    w_cfg: float = 1.0,
    device: str = 'cuda',
) -> dict:
    """
    Для каждого batch:
        1. Sample oracle embeddings (ODE).
        2. Retrieve top-K items.
        3. Для каждого user сравнить с target.
    
    Возвращает dict с метриками:
        {'HR@10': ..., 'NDCG@10': ..., 'HR@20': ..., 'NDCG@20': ...}
    """


def hit_ratio(rankings: torch.LongTensor, targets: torch.LongTensor, k: int) -> float:
    """
    rankings: (B, top_k) — predicted top-K item indices.
    targets:  (B,) — true target indices.
    Returns: HR@k = доля случаев, когда target в первых k.
    """


def ndcg(rankings: torch.LongTensor, targets: torch.LongTensor, k: int) -> float:
    """
    NDCG@k = sum(1 / log2(rank + 1)) для случаев, когда target в top-k.
    Усредняется по батчу.
    """