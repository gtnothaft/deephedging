"""GBM/Heston simulators and observable volatility features.

References
----------
[Heston1993] Heston, S. L. (1993), *A Closed-Form Solution for Options with
Stochastic Volatility with Applications to Bond and Currency Options*.

[RiskMetrics1996] J.P. Morgan/Reuters (1996), *RiskMetrics Technical Document*,
4th ed.
"""
from __future__ import annotations

import math
from typing import Tuple
import torch


def simulate_gbm(n_paths: int, cfg) -> torch.Tensor:
    r"""Simulate GBM stock paths with shape ``[n_paths, N + 1]``.

    .. math::
        S_{t+1}=S_t\exp\left[(\mu-\tfrac12\sigma^2)\Delta t
        +\sigma\sqrt{\Delta t}Z_t\right].
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    dt = cfg.T / cfg.N
    z = torch.randn(n_paths, cfg.N, device=cfg.device)
    increments = (
        (cfg.mu - 0.5 * cfg.sigma**2) * dt
        + cfg.sigma * math.sqrt(dt) * z
    )

    log_s = torch.empty(n_paths, cfg.N + 1, device=cfg.device)
    log_s[:, 0] = math.log(cfg.S0)
    log_s[:, 1:] = log_s[:, [0]] + torch.cumsum(increments, dim=1)
    return torch.exp(log_s)


def simulate_heston(
    n_paths: int,
    cfg,
    *,
    v0: float | None = None,
    kappa: float | None = None,
    theta: float | None = None,
    xi: float | None = None,
    rho: float | None = None,
    mu: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Simulate Heston paths using full-truncation Euler.

    The returned tensors have shape ``[n_paths, N + 1]``. The second tensor is
    the simulator's latent variance path. It is retained for the analytical
    Heston oracle benchmark, but it is **not** passed to the neural networks.

    Dynamics
    --------
    .. math::
        dS_t/S_t = \mu\,dt + \sqrt{v_t}\,dW_t^S,

    .. math::
        dv_t = \kappa(\theta-v_t)dt+\xi\sqrt{v_t}\,dW_t^v,
        \quad d\langle W^S,W^v\rangle_t=\rho\,dt.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")

    dt = cfg.T / cfg.N
    v0 = cfg.sigma**2 if v0 is None else v0
    kappa = cfg.heston_kappa if kappa is None else kappa
    theta = cfg.heston_theta if theta is None else theta
    xi = cfg.heston_xi if xi is None else xi
    rho = cfg.heston_rho if rho is None else rho
    mu = cfg.mu if mu is None else mu

    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1].")

    S = torch.empty((n_paths, cfg.N + 1), device=cfg.device)
    v = torch.empty((n_paths, cfg.N + 1), device=cfg.device)
    S[:, 0] = cfg.S0
    v[:, 0] = v0

    sqrt_one_minus_rho2 = math.sqrt(max(1.0 - rho**2, 0.0))

    for t in range(cfg.N):
        z_s = torch.randn(n_paths, device=cfg.device)
        z_v_independent = torch.randn(n_paths, device=cfg.device)
        z_v = rho * z_s + sqrt_one_minus_rho2 * z_v_independent

        v_pos = torch.clamp(v[:, t], min=0.0)
        S[:, t + 1] = S[:, t] * torch.exp(
            (mu - 0.5 * v_pos) * dt
            + torch.sqrt(v_pos * dt) * z_s
        )

        v_next = (
            v[:, t]
            + kappa * (theta - v_pos) * dt
            + xi * torch.sqrt(v_pos * dt) * z_v
        )
        v[:, t + 1] = torch.clamp(v_next, min=0.0)

    return S, v


def ewma_variance_proxy(
    S: torch.Tensor,
    cfg,
    *,
    decay: float | None = None,
) -> torch.Tensor:
    r"""Estimate annualized variance using only observed stock returns.

    The estimate is causal: ``v_hat[:, t]`` depends only on stock prices through
    time ``t``. The Heston neural networks receive this observable proxy rather
    than the simulator's latent variance.

    .. math::
        \hat v_t=\lambda\hat v_{t-1}
        +(1-\lambda)\frac{\log^2(S_t/S_{t-1})}{\Delta t}.

    The initial estimate is ``sigma**2``, interpreted as the volatility estimate
    available when the hedge is initiated.
    """
    if S.ndim != 2:
        raise ValueError("S must have shape [n_paths, N + 1].")
    if S.shape[1] != cfg.N + 1:
        raise ValueError(
            f"Expected {cfg.N + 1} time points, received {S.shape[1]}."
        )

    decay = cfg.ewma_decay if decay is None else float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError("EWMA decay must satisfy 0 <= decay < 1.")

    dt = cfg.T / cfg.N
    safe_s = torch.clamp(S, min=1e-12)
    log_returns = torch.diff(torch.log(safe_s), dim=1)

    v_hat = torch.empty_like(S)
    v_hat[:, 0] = cfg.sigma**2

    for t in range(1, cfg.N + 1):
        latest_annualized_squared_return = log_returns[:, t - 1] ** 2 / dt
        v_hat[:, t] = (
            decay * v_hat[:, t - 1]
            + (1.0 - decay) * latest_annualized_squared_return
        )

    return torch.clamp(v_hat, min=cfg.variance_floor)


def simulate_heston_with_observable_state(
    n_paths: int,
    cfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample Heston stock paths plus the observable EWMA variance proxy.

    The true variance is generated internally by the simulator and discarded
    before the sample is returned to the neural-network training loop.
    """
    S, _ = simulate_heston(n_paths, cfg)
    return S, ewma_variance_proxy(S, cfg)


def realized_volatility(S: torch.Tensor, T: float) -> torch.Tensor:
    """Return per-path annualized realized volatility from log returns."""
    log_ret = torch.diff(torch.log(torch.clamp(S, min=1e-12)), dim=1)
    return torch.sqrt(torch.sum(log_ret**2, dim=1) / T)
