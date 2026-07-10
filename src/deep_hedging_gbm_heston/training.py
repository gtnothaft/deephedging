r"""CVaR-trained neural-network hedge policies.

The deep-hedging framework allows the hedge policy to be optimized against a
monetary utility / risk measure [Buehler2019]. Here the primary objective is the
negative lower-tail CVaR of terminal net gain:

.. math::
    \mathcal L(\theta,w)
    =-w+\frac{1}{1-\alpha}\mathbb E[(w-G_\theta)^+],

with ``alpha = 0.50``. Minimizing jointly over policy parameters ``theta`` and
threshold ``w`` is equivalent to minimizing ``-CVaR_50%(G)``.
"""
from __future__ import annotations

import torch

from .black_scholes import initial_bs_price
from .pnl import terminal_pnl_policy


def _split_sampler_output(sample):
    """Return stock paths and optional observable extra state."""
    if isinstance(sample, tuple):
        if len(sample) < 2:
            raise ValueError("Tuple sampler output must contain at least two items.")
        return sample[0], sample[1]
    return sample, None


def negative_cvar_oce_loss(
    gain: torch.Tensor,
    threshold: torch.Tensor,
    confidence: float = 0.50,
) -> torch.Tensor:
    """Return the differentiable OCE representation of ``-CVaR(G)``.

    ``gain`` is terminal net hedging gain, so larger is better. At confidence
    0.50 the objective focuses on the average lower half of the gain
    distribution. Lower values of this loss are better.
    """
    if gain.ndim != 1:
        raise ValueError("gain must be a one-dimensional tensor.")
    if not 0.0 <= confidence < 1.0:
        raise ValueError("confidence must satisfy 0 <= confidence < 1.")

    tail_multiplier = 1.0 / (1.0 - confidence)
    shortfall = torch.relu(threshold - gain)
    return -threshold + tail_multiplier * torch.mean(shortfall)


def train_policy(
    policy,
    cfg,
    sampler,
    *,
    name: str = "policy",
    initial_premium: float | None = None,
    verbose: bool = True,
):
    """Train a hedge policy by minimizing ``-CVaR_50%(G)``.

    Returns
    -------
    list[float]
        Loss history. The learned auxiliary CVaR threshold is stored as
        ``policy.cvar_threshold_`` for diagnostics.
    """
    policy.to(cfg.device)
    torch.manual_seed(cfg.seed)

    if initial_premium is None:
        initial_premium = initial_bs_price(cfg)

    cvar_threshold = torch.nn.Parameter(
        torch.tensor(0.0, device=cfg.device)
    )
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + [cvar_threshold],
        lr=cfg.lr,
    )
    history: list[float] = []

    for epoch in range(1, cfg.epochs + 1):
        sample = sampler(cfg.batch_size, cfg)
        S, observable_state = _split_sampler_output(sample)

        gain = terminal_pnl_policy(
            policy,
            S,
            cfg,
            initial_premium,
            extra_state=observable_state,
        )
        loss = negative_cvar_oce_loss(
            gain,
            cvar_threshold,
            confidence=cfg.cvar_confidence,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.parameters(), max_norm=cfg.grad_clip
        )
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        history.append(loss_value)

        if verbose and (
            epoch == 1 or epoch % 50 == 0 or epoch == cfg.epochs
        ):
            threshold_value = float(cvar_threshold.detach().cpu())
            print(
                f"{name:22s} epoch {epoch:4d}/{cfg.epochs}, "
                f"-CVaR loss={loss_value:.6f}, "
                f"threshold={threshold_value:.6f}"
            )

    policy.cvar_threshold_ = float(cvar_threshold.detach().cpu())
    return history


def train_policy_with_sampler(*args, **kwargs):
    """Backward-compatible alias."""
    return train_policy(*args, **kwargs)
