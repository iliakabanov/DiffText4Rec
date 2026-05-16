@dataclass
class TrainConfig:
    """Все гиперпараметры в одном месте."""
    # Data
    data_dir: str
    padding_token_id: int
    batch_size: int = 256
    num_workers: int = 4
    
    # Model
    d_cond: int = 512
    encoder_layers: int = 2
    encoder_heads: int = 8
    denoiser_blocks: int = 6
    denoiser_d_block: int = 1024
    denoiser_d_ff: int = 2048
    
    # Training
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    num_steps: int = 100_000
    p_uncond: float = 0.1
    
    # EMA
    ema_decay: float = 0.9999
    
    # Eval
    eval_every: int = 5000
    sample_num_steps: int = 50
    w_cfg: float = 1.0
    top_k_values: tuple = (10, 20)
    
    # IO
    output_dir: str = './checkpoints'


class EMA:
    """
    Exponential Moving Average параметров модели.
    Хранит копию весов, обновляет на каждом training step.
    Для sampling используем именно EMA-веса — это критично для качества.
    """
    
    def __init__(self, model: nn.Module, decay: float):
        ...
    
    def update(self, model: nn.Module):
        """Обновляет EMA после optimizer.step()."""
    
    def copy_to(self, model: nn.Module):
        """Копирует EMA-веса в model (для eval)."""
    
    def restore(self, model: nn.Module):
        """Возвращает исходные веса после eval."""


def get_lr_schedule(optimizer, warmup_steps, total_steps):
    """LambdaLR: linear warmup, потом константа (или cosine decay)."""


def train(config: TrainConfig):
    """
    Главная функция. Цикл:
        for step in 1..num_steps:
            batch = next(train_loader)
            loss = flow_matcher.compute_loss(...)
            loss.backward()
            clip_grad_norm
            optimizer.step()
            ema.update(model)
            
            if step % eval_every == 0:
                evaluate_on_val(...)
                save_checkpoint(...)
    """