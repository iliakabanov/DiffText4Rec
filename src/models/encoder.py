class HistoryEncoder(nn.Module):
    """
    Transformer encoder:
        history_indices (B, L) -> conditioning (B, D_cond)
    
    Архитектура:
        1. Embedding lookup (frozen TEM embeddings, whitened) -> (B, L, D_emb)
        2. Linear projection D_emb -> D_hidden -> (B, L, D_hidden)
        3. + positional embedding (learnable, длиной max_seq_len)
        4. N transformer encoder layers (self-attention + FFN)
        5. Last-item Pooling -> (B, D_hidden)
        6. Output projection D_hidden -> D_cond -> (B, D_cond)
    """
    
    def __init__(
        self,
        item_embeddings: torch.Tensor,    # (N_items + 1, D_emb), последняя строка — padding
        padding_token_id: int,
        d_hidden: int = 512,
        d_cond: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 10,
        pool: str = 'mean',  # 'mean' | 'last' | 'cls'
    ):
        ...
    
    def forward(
        self,
        history: torch.LongTensor,  # (B, L)
        mask: torch.FloatTensor,    # (B, L)
    ) -> torch.Tensor:
        """Возвращает conditioning vector (B, D_cond)."""