"""Black-Scholes price and delta utilities.

Reference: Black and Scholes (1973), The Pricing of Options and Corporate Liabilities.
"""
import math
from typing import Tuple
import torch

_NORMAL = torch.distributions.Normal(0.0, 1.0)


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard-normal CDF, vectorized for torch tensors."""
    return _NORMAL.cdf(x)


def bs_call_price_delta(
    S: torch.Tensor,
    K: float,
    tau: torch.Tensor,
    sigma: float | torch.Tensor,
    r: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Black-Scholes European call price and delta.

    Parameters
    ----------
    S:
        Spot price tensor.
    K:
        Strike.
    tau:
        Time to maturity in years. Same shape as S or broadcastable.
    sigma:
        Volatility. Can be a scalar or a tensor broadcastable with S.
    r:
        Continuously compounded risk-free rate.

    Returns
    -------
    price, delta:
        Tensors with the same shape as S.
    """
    eps = torch.tensor(1e-8, device=S.device, dtype=S.dtype)
    tau_safe = torch.clamp(tau, min=eps)

    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=S.device, dtype=S.dtype)
    sigma = torch.clamp(sigma, min=eps)

    vol_sqrt = sigma * torch.sqrt(tau_safe)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau_safe) / vol_sqrt
    d2 = d1 - vol_sqrt

    price = S * norm_cdf(d1) - K * torch.exp(-r * tau_safe) * norm_cdf(d2)
    delta = norm_cdf(d1)

    # Safe maturity behavior.
    price = torch.where(tau <= eps, torch.clamp(S - K, min=0.0), price)
    delta = torch.where(tau <= eps, (S > K).float(), delta)
    return price, delta


def initial_bs_price(cfg) -> float:
    """Initial Black-Scholes premium for the configured call."""
    S0 = torch.tensor(float(cfg.S0), device=cfg.device)
    tau = torch.tensor(float(cfg.T), device=cfg.device)
    price, _ = bs_call_price_delta(S0, cfg.K, tau, cfg.sigma, cfg.r)
    return float(price.detach().cpu())
