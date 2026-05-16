class FlowMatcher(nn.Module):
    """
    Conditional Flow Matching с прямой траекторией (rectified flow).
    
    Forward (для training):
        history -> encoder -> c
        target  -> z_1 (через embedding lookup, whitened)
        z_0 ~ N(0, I)
        τ ~ Uniform(0, 1) или logit-normal
        z_τ = (1 - τ) z_0 + τ z_1
        v_true = z_1 - z_0
        v_pred = denoiser(z_τ, τ, c)
        loss = MSE(v_pred, v_true)
    
    Sample:
        z_0 ~ N(0, I)
        Euler ODE с шагом dt: z_{τ+dt} = z_τ + dt * v_pred(z_τ, τ, c)
        с classifier-free guidance: v = (1+w) v_cond - w v_uncond
    """
    
    def __init__(
        self,
        encoder: HistoryEncoder,
        denoiser: MLPDenoiser,
        item_embeddings_whitened: torch.Tensor,  # (N_items + 1, D_emb)
        padding_token_id: int,
        p_uncond: float = 0.1,
        tau_distribution: str = 'uniform',  # 'uniform' | 'logit_normal'
    ):
        ...
    
    def compute_loss(
        self,
        history: torch.LongTensor,
        target: torch.LongTensor,
        mask: torch.FloatTensor,
    ) -> torch.Tensor:
        """Возвращает скалярный loss. CFG dropout применяется здесь."""
    
    @torch.no_grad()
    def sample(
        self,
        history: torch.LongTensor,
        mask: torch.FloatTensor,
        num_steps: int = 50,
        w_cfg: float = 1.0,
    ) -> torch.Tensor:
        """
        Генерирует oracle embeddings (B, D_emb).
        Использует Euler ODE с CFG.
        """
    
    @torch.no_grad()
    def retrieve(
        self,
        oracle_embeddings: torch.Tensor,  # (B, D_emb)
        top_k: int = 20,
    ) -> torch.Tensor:
        """
        Скалярное произведение с item embedding matrix.
        Возвращает (B, top_k) — индексы топ-K товаров.
        Padding token из retrieval исключается.
        """