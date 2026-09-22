"""
Генеративные процессы над эмбеддингами товаров. Интерфейс у обоих одинаковый:

    loss(denoiser, x_clean, h)                -> скаляр (обучение)
    sample(denoiser, h, num_steps, w_cfg)     -> (B, d) oracle-эмбеддинг (инференс)

Денойзер получает время как долю пути от шума к данным τ ∈ [0, 1] (0 — шум, 1 — товар).

    DDPM          x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε, t ∈ {0..T−1}; сеть предсказывает x_0;
                  сэмплинг — ancestral DDPM по (прорежённой) сетке шагов.
    FlowMatching  z_τ = (1−τ)·z_0 + τ·z_1, z_0 ~ N(0, I); сеть предсказывает скорость
                  u = z_1 − z_0 (prediction='velocity') или z_1 (prediction='x1');
                  сэмплинг — Euler ODE от τ=0 к τ=1.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import ConcatDenoiser


# --------------------------------------------------------------------------- #
# Расписания шума DDPM
# --------------------------------------------------------------------------- #

def _exp_schedule(T: int, beta_min: float = 0.1, beta_max: float = 10.0) -> torch.Tensor:
    """Экспоненциальное расписание (дефолт DreamRec)."""
    t = torch.arange(1, T + 1, dtype=torch.float64)
    betas = 1.0 - torch.exp(-beta_min / T - t / T ** 2 * (beta_max - beta_min))
    return betas.clamp(0.0, 0.999).float()


def _linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def _cosine_schedule(T: int) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_cumprod = (f / f[0]).clamp(min=1e-8)
    return (1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(0.0, 0.999).float()


BETA_SCHEDULES = {'exp': _exp_schedule, 'linear': _linear_schedule, 'cosine': _cosine_schedule}


# --------------------------------------------------------------------------- #
# DDPM
# --------------------------------------------------------------------------- #

class DDPM(nn.Module):
    """DDPM с x_0-параметризацией (как в DreamRec)."""

    def __init__(self, T: int, beta_schedule: str):
        super().__init__()
        if beta_schedule not in BETA_SCHEDULES:
            raise ValueError(f'beta_schedule должен быть одним из {list(BETA_SCHEDULES)}, получено {beta_schedule!r}')
        self.T = T
        self.beta_schedule = beta_schedule
        betas = BETA_SCHEDULES[beta_schedule](T)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1.0 - betas, dim=0))

    def tau(self, t: torch.Tensor) -> torch.Tensor:
        """Шаг t (t = 0 — почти чистый x) → доля пути от шума к данным τ = 1 − t/T."""
        return 1.0 - t.float() / self.T

    def loss(self, denoiser: ConcatDenoiser, x_clean: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """L = E_{t,ε} ‖x_0 − f(√ᾱ_t·x_0 + √(1−ᾱ_t)·ε, h, τ(t))‖²"""
        t   = torch.randint(0, self.T, (x_clean.shape[0],), device=x_clean.device)
        eps = torch.randn_like(x_clean)
        ab  = self.alphas_cumprod[t].unsqueeze(-1)
        x_t = ab.sqrt() * x_clean + (1.0 - ab).sqrt() * eps
        return F.mse_loss(denoiser(x_t, h, self.tau(t)), x_clean)

    @torch.no_grad()
    def sample(
        self,
        denoiser: ConcatDenoiser,
        h: torch.Tensor,
        num_steps: int | None = None,
        w_cfg: float = 0.0,
    ) -> torch.Tensor:
        """
        Ancestral DDPM по num_steps равномерно прорежённым шагам (None или ≥ T — по всем T).
        Posterior-коэффициенты пересчитываются под сетку; на полной сетке это обычный DDPM.
        """
        n = self.T if num_steps is None else min(num_steps, self.T)
        S = torch.unique(torch.linspace(0, self.T - 1, n, device=h.device).round().long())

        ac      = self.alphas_cumprod[S]
        ac_prev = torch.cat([ac.new_ones(1), ac[:-1]])
        betas   = 1.0 - ac / ac_prev
        coef_x0 = ac_prev.sqrt() * betas / (1.0 - ac)
        coef_xt = (1.0 - betas).sqrt() * (1.0 - ac_prev) / (1.0 - ac)
        sigma   = ((1.0 - ac_prev) / (1.0 - ac) * betas).clamp(min=1e-20).sqrt()

        x = torch.randn_like(h)
        for i in reversed(range(len(S))):
            tau = self.tau(S[i].expand(h.shape[0]))
            x0  = denoiser.guided(x, h, tau, w_cfg)
            x   = coef_x0[i] * x0 + coef_xt[i] * x
            if i > 0:
                x = x + sigma[i] * torch.randn_like(x)
        return x

    def config(self) -> dict:
        return {'process': 'ddpm', 'T': self.T, 'beta_schedule': self.beta_schedule}


# --------------------------------------------------------------------------- #
# Flow Matching
# --------------------------------------------------------------------------- #

FM_PREDICTIONS = ('velocity', 'x1')


class FlowMatching(nn.Module):
    """
    Conditional Flow Matching (rectified flow). Что предсказывает сеть — задаёт prediction:

        velocity  f ≈ u = z_1 − z_0;              скорость для ODE — сам выход сети
        x1        f ≈ z_1 (эмбеддинг товара);      скорость для ODE — (f − z_τ) / (1 − τ)

    При точном предсказании обе дают одно и то же поле скоростей; различаются loss
    (ошибка в x1 весит в скорости как 1/(1 − τ)) и тем, что учит сеть.
    """

    DEFAULT_STEPS = 50

    def __init__(self, prediction: str = 'velocity'):
        super().__init__()
        if prediction not in FM_PREDICTIONS:
            raise ValueError(f'prediction должен быть одним из {FM_PREDICTIONS}, получено {prediction!r}')
        self.prediction = prediction

    def loss(self, denoiser: ConcatDenoiser, x_clean: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        z_1 = x_clean, z_τ = (1−τ)·z_0 + τ·z_1:
            velocity  L = E_{τ,z_0} ‖(z_1 − z_0) − f(z_τ, h, τ)‖²
            x1        L = E_{τ,z_0} ‖z_1 − f(z_τ, h, τ)‖²
        """
        z0    = torch.randn_like(x_clean)
        tau   = torch.rand(x_clean.shape[0], device=x_clean.device)
        z_tau = (1.0 - tau[:, None]) * z0 + tau[:, None] * x_clean
        target = x_clean - z0 if self.prediction == 'velocity' else x_clean
        return F.mse_loss(denoiser(z_tau, h, tau), target)

    @torch.no_grad()
    def sample(
        self,
        denoiser: ConcatDenoiser,
        h: torch.Tensor,
        num_steps: int | None = None,
        w_cfg: float = 0.0,
    ) -> torch.Tensor:
        """
        Euler ODE от шума (τ=0) к эмбеддингу товара (τ=1) за num_steps шагов.
        τ шага не больше 1 − 1/num_steps, так что деление на 1 − τ при x1 безопасно.
        """
        n  = num_steps or self.DEFAULT_STEPS
        dt = 1.0 / n
        z  = torch.randn_like(h)
        for i in range(n):
            tau = torch.full((h.shape[0],), i * dt, device=h.device)
            z   = z + dt * self.velocity(denoiser, z, h, tau, w_cfg)
        return z

    @torch.no_grad()
    def velocity(self, denoiser: ConcatDenoiser, z: torch.Tensor, h: torch.Tensor,
                 tau: torch.Tensor, w_cfg: float = 0.0) -> torch.Tensor:
        """Скорость ODE в z_τ: выход сети (velocity) или (x̂_1 − z_τ) / (1 − τ) (x1)."""
        out = denoiser.guided(z, h, tau, w_cfg)
        return out if self.prediction == 'velocity' else (out - z) / (1.0 - tau[:, None])

    def config(self) -> dict:
        return {'process': 'flow_matching', 'prediction': self.prediction}
