r"""Hedging gain, transaction-cost, and turnover utilities.

The terminal net gain convention is

.. math::
    G=V_0-(S_T-K)^+ + \sum_t \delta_t(S_{t+1}-S_t)-C_T,

with proportional stock-trading costs

.. math::
    C_T=\gamma\sum_t S_t|\delta_t-\delta_{t-1}|
    +\gamma S_T|\delta_{N-1}|.

The final term liquidates the remaining stock hedge at maturity. This is a
stock-only specialization of the transaction-cost treatment in [Buehler2019].
"""
from __future__ import annotations

import torch

from .black_scholes import bs_call_price_delta
from .heston import heston_delta_torch, initial_heston_price


def _validate_path_shapes(S: torch.Tensor, cfg) -> None:
    if S.ndim != 2 or S.shape[1] != cfg.N + 1:
        raise ValueError(
            f"S must have shape [n_paths, {cfg.N + 1}], received {tuple(S.shape)}."
        )


def _policy_delta(
    policy,
    t: int,
    S_t: torch.Tensor,
    previous_delta: torch.Tensor,
    extra_state: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate a policy using only the state it is allowed to observe."""
    if getattr(policy, "uses_extra_state", False):
        if extra_state is None:
            raise ValueError("This policy requires an observable extra-state path.")
        return policy.forward_one_step(
            t,
            S_t,
            previous_delta,
            extra_state[:, t],
        )
    return policy.forward_one_step(t, S_t, previous_delta)


def terminal_pnl_policy(
    policy,
    S: torch.Tensor,
    cfg,
    initial_premium: float,
    *,
    extra_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return terminal net gain for a short call hedged by an NN policy."""
    gain, _, _ = pnl_cost_turnover_policy(
        policy,
        S,
        cfg,
        initial_premium,
        extra_state=extra_state,
    )
    return gain


def pnl_cost_turnover_policy(
    policy,
    S: torch.Tensor,
    cfg,
    initial_premium: float,
    *,
    extra_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return terminal gain, total cost, and turnover for an NN policy."""
    _validate_path_shapes(S, cfg)
    if extra_state is not None and extra_state.shape != S.shape:
        raise ValueError("extra_state must have the same shape as S.")

    n_paths = S.shape[0]
    device = S.device
    gain = torch.full(
        (n_paths,), float(initial_premium), device=device, dtype=S.dtype
    )
    previous_delta = torch.zeros(n_paths, device=device, dtype=S.dtype)
    total_cost = torch.zeros_like(previous_delta)
    turnover = torch.zeros_like(previous_delta)

    for t in range(cfg.N):
        S_t = S[:, t]
        delta = _policy_delta(policy, t, S_t, previous_delta, extra_state)
        trade = delta - previous_delta

        turnover = turnover + torch.abs(trade)
        total_cost = total_cost + cfg.cost_gamma * S_t * torch.abs(trade)
        gain = gain + delta * (S[:, t + 1] - S_t)
        previous_delta = delta

    # Liquidate the remaining stock position at maturity.
    turnover = turnover + torch.abs(previous_delta)
    total_cost = (
        total_cost
        + cfg.cost_gamma * S[:, -1] * torch.abs(previous_delta)
    )

    payoff = torch.clamp(S[:, -1] - cfg.K, min=0.0)
    gain = gain - payoff - total_cost
    return gain, total_cost, turnover


def pnl_cost_turnover_from_delta_path(
    S: torch.Tensor,
    delta_path: torch.Tensor,
    cfg,
    initial_premium: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate a precomputed stock-position path under the configured cost.

    Parameters
    ----------
    delta_path:
        Stock positions at the ``N`` hedge dates, shape ``[n_paths, N]``.
    """
    _validate_path_shapes(S, cfg)
    if delta_path.shape != (S.shape[0], cfg.N):
        raise ValueError(
            f"delta_path must have shape {(S.shape[0], cfg.N)}, "
            f"received {tuple(delta_path.shape)}."
        )

    previous = torch.zeros_like(delta_path[:, 0])
    total_cost = torch.zeros_like(previous)
    turnover = torch.zeros_like(previous)
    trading_gain = torch.zeros_like(previous)

    for t in range(cfg.N):
        delta = delta_path[:, t]
        trade = delta - previous
        turnover = turnover + torch.abs(trade)
        total_cost = total_cost + cfg.cost_gamma * S[:, t] * torch.abs(trade)
        trading_gain = trading_gain + delta * (S[:, t + 1] - S[:, t])
        previous = delta

    turnover = turnover + torch.abs(previous)
    total_cost = total_cost + cfg.cost_gamma * S[:, -1] * torch.abs(previous)

    payoff = torch.clamp(S[:, -1] - cfg.K, min=0.0)
    gain = float(initial_premium) - payoff + trading_gain - total_cost
    return gain, total_cost, turnover


def bs_delta_path(S: torch.Tensor, cfg) -> torch.Tensor:
    """Return analytical Black-Scholes stock deltas at every hedge date."""
    _validate_path_shapes(S, cfg)
    deltas = []
    for t in range(cfg.N):
        S_t = S[:, t]
        tau = torch.full_like(S_t, cfg.T * (cfg.N - t) / cfg.N)
        _, delta = bs_call_price_delta(S_t, cfg.K, tau, cfg.sigma, cfg.r)
        deltas.append(delta)
    return torch.stack(deltas, dim=1)


def pnl_cost_turnover_bs_delta(
    S: torch.Tensor,
    cfg,
    initial_premium: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate analytical Black-Scholes delta hedging in the GBM section."""
    return pnl_cost_turnover_from_delta_path(
        S,
        bs_delta_path(S, cfg),
        cfg,
        initial_premium,
    )


def heston_delta_path(
    S: torch.Tensor,
    v: torch.Tensor,
    cfg,
    *,
    n_quad: int = 64,
    chunk_size: int = 20_000,
) -> torch.Tensor:
    """Return model-consistent Heston stock deltas at all hedge dates.

    The baseline uses the simulator's true latent variance. It is therefore an
    oracle, model-consistent benchmark rather than an implementable data-only
    strategy.
    """
    _validate_path_shapes(S, cfg)
    if v.shape != S.shape:
        raise ValueError("v must have the same shape as S.")

    deltas = []
    for t in range(cfg.N):
        tau = cfg.T * (cfg.N - t) / cfg.N
        deltas.append(
            heston_delta_torch(
                S[:, t],
                v[:, t],
                tau,
                cfg,
                n_quad=n_quad,
                chunk_size=chunk_size,
            )
        )
    return torch.stack(deltas, dim=1)


def pnl_cost_turnover_heston_delta(
    S: torch.Tensor,
    v: torch.Tensor,
    cfg,
    initial_premium: float | None = None,
    *,
    delta_path: torch.Tensor | None = None,
    n_quad: int = 64,
    chunk_size: int = 20_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the analytical Heston stock-delta oracle benchmark."""
    if initial_premium is None:
        initial_premium = initial_heston_price(cfg)
    if delta_path is None:
        delta_path = heston_delta_path(
            S,
            v,
            cfg,
            n_quad=n_quad,
            chunk_size=chunk_size,
        )
    return pnl_cost_turnover_from_delta_path(
        S,
        delta_path,
        cfg,
        initial_premium,
    )
