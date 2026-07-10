r"""Out-of-sample hedging metrics.

The primary project score is

.. math::
    \mathrm{Score}=-\mathrm{CVaR}_{50\%}(G),

where ``G`` is raw terminal net gain after payoff and transaction costs. Lower
scores are better.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


SCORE_COLUMN = "Score = -CVaR50%(G)"


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def empirical_cvar_gain(
    gain,
    confidence: float = 0.50,
) -> float:
    """Return empirical lower-tail CVaR of terminal gain.

    At ``confidence=0.50``, this is the arithmetic mean of the lowest 50% of
    observations. All observations are used for scoring; no plot clipping or
    mean adjustment is applied.
    """
    if not 0.0 <= confidence < 1.0:
        raise ValueError("confidence must satisfy 0 <= confidence < 1.")

    values = _to_numpy(gain).astype(float, copy=False).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("gain must contain at least one finite value.")

    tail_fraction = 1.0 - confidence
    n_tail = max(1, int(np.ceil(tail_fraction * values.size)))
    worst_tail = np.partition(values, n_tail - 1)[:n_tail]
    return float(np.mean(worst_tail))


def negative_cvar_score(
    gain,
    confidence: float = 0.50,
) -> float:
    """Return ``Score = -CVaR_confidence(G)``; lower is better."""
    return -empirical_cvar_gain(gain, confidence=confidence)


def risk_metrics(
    gain,
    cost=None,
    turnover=None,
    *,
    cvar_confidence: float = 0.50,
) -> dict:
    """Compute the score and supporting out-of-sample hedging metrics."""
    values = _to_numpy(gain).astype(float, copy=False).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("gain must contain at least one finite value.")

    row = {
        SCORE_COLUMN: negative_cvar_score(
            values, confidence=cvar_confidence
        ),
        "CVaR50% Gain": empirical_cvar_gain(
            values, confidence=cvar_confidence
        ),
        "Mean P&L": float(np.mean(values)),
        "Std P&L": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "RMSE": float(np.sqrt(np.mean(values**2))),
    }

    if cost is not None:
        row["Avg Cost"] = float(np.mean(_to_numpy(cost)))
    if turnover is not None:
        row["Avg Turnover"] = float(np.mean(_to_numpy(turnover)))
    return row


def comparison_table(
    results: dict,
    *,
    cvar_confidence: float = 0.50,
) -> pd.DataFrame:
    """Create a strategy table sorted from best to worst primary score."""
    rows = []
    for strategy, values in results.items():
        if len(values) != 3:
            raise ValueError(
                "Each result must be a (gain, cost, turnover) tuple."
            )
        gain, cost, turnover = values
        row = risk_metrics(
            gain,
            cost,
            turnover,
            cvar_confidence=cvar_confidence,
        )
        row["Strategy"] = strategy
        rows.append(row)

    table = pd.DataFrame(rows).set_index("Strategy")
    return table.sort_values(SCORE_COLUMN, ascending=True)


def cost_sweep_table(
    experiments: dict[float, dict],
    *,
    cvar_confidence: float = 0.50,
) -> pd.DataFrame:
    """Combine Heston results from several transaction-cost levels."""
    tables = []
    for gamma, experiment in experiments.items():
        table = comparison_table(
            experiment["results"],
            cvar_confidence=cvar_confidence,
        ).reset_index()
        table.insert(0, "cost_gamma", float(gamma))
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
