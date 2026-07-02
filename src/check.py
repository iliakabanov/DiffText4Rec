import torch
import pickle
import numpy as np
from train_fm_text import TrainConfig
from data import build_dataloaders
from models.flow_matcher import FlowMatcher

cfg = TrainConfig()
device = torch.device('cuda')

loaders = build_dataloaders(cfg.data_dir, batch_size=8, num_workers=0)
batch = next(iter(loaders['val']))
history = batch['history'].to(device)
mask    = batch['mask'].to(device)
target  = batch['target'].to(device)

with open(cfg.emb_path, 'rb') as f:
    emb_raw = pickle.load(f).astype(np.float32)

model = FlowMatcher(
    item_embeddings_raw=emb_raw,
    padding_idx=cfg.padding_idx,
    n_components=cfg.n_components,
    whitening_cache_path=cfg.whitening_cache,
    d_hidden=cfg.d_hidden, d_cond=cfg.d_cond,
    encoder_layers=cfg.encoder_layers,
    encoder_heads=cfg.encoder_heads,
    encoder_d_ff=cfg.encoder_d_ff,
    encoder_dropout=cfg.encoder_dropout,
    denoiser_d_block=cfg.denoiser_d_block,
    denoiser_d_ff=cfg.denoiser_d_ff,
    denoiser_n_blocks=cfg.denoiser_n_blocks,
    denoiser_time_emb_dim=cfg.denoiser_time_emb_dim,
    p_uncond=cfg.p_uncond,
).to(device)

ckpt = torch.load('checkpoints/baseline/best.pt', map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()

item_emb = model.encoder.item_embedding.weight[:model.n_items]
B = history.shape[0]
torch.manual_seed(42)

with torch.no_grad():
    z_1 = model.encoder.item_embedding(target)
    z_0 = torch.randn_like(z_1)
    tau = torch.rand(B, device=device)
    z_tau = (1 - tau.view(B,1)) * z_0 + tau.view(B,1) * z_1
    u_true = z_1 - z_0
    c = model.encoder(history, mask)
    v_pred = model.denoiser(z_tau, tau, c)

    # 1. Ключевые нормы
    print(f'z_1 norm:    {z_1.norm(dim=-1).mean():.4f}  (expected ~{cfg.n_components**0.5:.1f})')
    print(f'z_0 norm:    {z_0.norm(dim=-1).mean():.4f}  (expected ~{cfg.n_components**0.5:.1f})')
    print(f'u_true norm: {u_true.norm(dim=-1).mean():.4f}  (expected ~{(2*cfg.n_components)**0.5:.1f})')
    print(f'v_pred norm: {v_pred.norm(dim=-1).mean():.4f}')

    # 2. Cosine similarity v_pred vs u_true
    vp_n = v_pred / v_pred.norm(dim=-1, keepdim=True)
    ut_n = u_true / u_true.norm(dim=-1, keepdim=True)
    cos_vu = (vp_n * ut_n).sum(dim=-1)
    print(f'\nCos sim v_pred vs u_true: mean={cos_vu.mean():.4f}  (was 0.547 в 3072d)')

    # 3. Unconditional тоже
    phi = model.phi.unsqueeze(0).expand(B, -1)
    v_uncond = model.denoiser(z_tau, tau, phi)
    vun_n = v_uncond / v_uncond.norm(dim=-1, keepdim=True)
    cos_uncond = (vun_n * ut_n).sum(dim=-1)
    print(f'Cos sim v_uncond vs u_true: mean={cos_uncond.mean():.4f}')

    # 4. Oracle vs target
    oracle = model.sample(history, mask, num_steps=50, w_cfg=0.0)
    print(f'\noracle norm: {oracle.norm(dim=-1).mean():.4f}')
    on = oracle / oracle.norm(dim=-1, keepdim=True)
    z1n = z_1 / z_1.norm(dim=-1, keepdim=True)
    cos_ot = (on * z1n).sum(dim=-1)
    print(f'Cos sim oracle vs target: mean={cos_ot.mean():.4f}')

    # 5. Ранги
    scores = oracle @ item_emb.T
    print(f'\nRanks (w_cfg=0.0):')
    for i in range(8):
        ts = scores[i, target[i].item()].item()
        rank = (scores[i] > ts).sum().item() + 1
        print(f'  user {i}: rank={rank}/{model.n_items}')

    # 6. Variance explained
    print(f'\nPCA n_components: {cfg.n_components}')
    print(f'whitening_V shape: {model.whitening_V.shape}')

torch.manual_seed(42)
with torch.no_grad():
    oracle = model.sample(history, mask, num_steps=50, w_cfg=0.0)
    
    # Вариант 1: без нормализации
    scores1 = oracle @ item_emb.T
    ranks1 = [(scores1[i] > scores1[i, target[i].item()]).sum().item() + 1 
              for i in range(B)]
    print(f'Without norm: ranks={ranks1}, mean={sum(ranks1)/len(ranks1):.0f}')
    
    # Вариант 2: per-sample нормализация (не меняет порядок!)
    target_norm = item_emb.norm(dim=-1).mean()
    oracle_n = oracle * (target_norm / oracle.norm(dim=-1, keepdim=True))
    scores2 = oracle_n @ item_emb.T
    ranks2 = [(scores2[i] > scores2[i, target[i].item()]).sum().item() + 1 
              for i in range(B)]
    print(f'Per-sample norm: ranks={ranks2}, mean={sum(ranks2)/len(ranks2):.0f}')
    
    # Проверяем что направления совпадают
    cos = (oracle / oracle.norm(dim=-1, keepdim=True) * 
           oracle_n / oracle_n.norm(dim=-1, keepdim=True)).sum(dim=-1)
    print(f'Direction preserved: {cos}  (должно быть 1.0)')
    
    # Вариант 3: попробуем разные w_cfg
    for w in [0.0, 0.5, 1.0, -0.5]:
        torch.manual_seed(42)
        oracle_w = model.sample(history, mask, num_steps=50, w_cfg=w)
        scores_w = oracle_w @ item_emb.T
        ranks_w = [(scores_w[i] > scores_w[i, target[i].item()]).sum().item() + 1
                   for i in range(B)]
        print(f'w={w:5.1f}: mean rank={sum(ranks_w)/len(ranks_w):.0f}, ranks={ranks_w}')