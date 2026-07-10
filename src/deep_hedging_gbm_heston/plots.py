"""Plotting utilities for the GBM/Heston deep-hedging experiments."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch

from .metrics import negative_cvar_score


def _to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_pnl_distributions(
    pnl_by_label: dict,
    *,
    adjust_mean: bool = False,
    title: str = "Terminal P&L distributions",
    bins: int = 80,
    save_path: str | Path | None = None,
):
    """Plot terminal P&L distributions as histogram outlines.

    The final Heston cost-sweep figure should use
    :func:`plot_heston_cost_sweep_density`, which plots raw net P&L and reports
    the primary CVaR score.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, pnl in pnl_by_label.items():
        values = _to_numpy(pnl).reshape(-1)
        if adjust_mean:
            values = values - values.mean()
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=label,
        )

    ax.axvline(0, linestyle="--", linewidth=1, color="0.45")
    ax.set_xlabel("Adjusted P&L" if adjust_mean else "Terminal net P&L, G")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250, bbox_inches="tight", facecolor="white")
    return ax


def _safe_kde(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Evaluate a KDE, adding negligible jitter only if a sample is singular."""
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite observations are required for KDE.")
    if np.std(values) < 1e-12:
        values = values + np.linspace(-1e-6, 1e-6, values.size)
    return gaussian_kde(values, bw_method="scott")(x_grid)


def plot_heston_cost_sweep_density(
    cost_sweep_results: dict[float, dict],
    *,
    confidence: float = 0.50,
    output_png: str | Path | None = None,
    output_pdf: str | Path | None = None,
):
    """Create the final three-panel Heston P&L density figure.

    Each panel corresponds to one transaction-cost level and compares exactly
    three strategies: analytical Heston delta, Simple NN, and Recurrent NN.
    Curves use **raw terminal net P&L**, not mean-adjusted P&L. The displayed
    scores are computed from all out-of-sample observations.
    """
    strategy_order = [
        "Analytical Heston delta",
        "Simple NN",
        "Recurrent NN",
    ]
    colors = {
        "Analytical Heston delta": "#202124",
        "Simple NN": "#0072B2",
        "Recurrent NN": "#D55E00",
    }
    gammas = sorted(float(g) for g in cost_sweep_results)
    if len(gammas) != 3:
        raise ValueError("The presentation figure expects exactly three cost levels.")

    all_values = []
    for gamma in gammas:
        results = cost_sweep_results[gamma]["results"]
        missing = set(strategy_order) - set(results)
        if missing:
            raise KeyError(f"Missing strategies for gamma={gamma}: {sorted(missing)}")
        for strategy in strategy_order:
            all_values.append(_to_numpy(results[strategy][0]).reshape(-1))

    pooled = np.concatenate(all_values)
    pooled = pooled[np.isfinite(pooled)]
    # Clip only the plotted x-range, not the data used for scores.
    lower, upper = np.quantile(pooled, [0.001, 0.999])
    span = max(upper - lower, 1e-6)
    x_grid = np.linspace(lower - 0.05 * span, upper + 0.05 * span, 700)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.6, 4.9),
        sharex=True,
        sharey=True,
    )

    for ax, gamma in zip(axes, gammas):
        results = cost_sweep_results[gamma]["results"]
        for strategy in strategy_order:
            gain = _to_numpy(results[strategy][0]).astype(float).reshape(-1)
            gain = gain[np.isfinite(gain)]
            density = _safe_kde(gain, x_grid)
            score = negative_cvar_score(gain, confidence=confidence)
            label = f"{strategy}\nScore = {score:.3f}"

            ax.plot(
                x_grid,
                density,
                color=colors[strategy],
                linewidth=2.35,
                label=label,
            )
            ax.fill_between(
                x_grid,
                0.0,
                density,
                color=colors[strategy],
                alpha=0.07,
            )

        ax.axvline(0.0, color="0.55", linestyle="--", linewidth=1.0)
        ax.set_title(
            rf"Transaction cost $\gamma={gamma:g}$",
            fontsize=13,
            fontweight="semibold",
            pad=10,
        )
        ax.set_xlabel("Terminal net P&L, $G$", fontsize=11)
        ax.grid(axis="y", alpha=0.16, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.4, loc="upper left")

    axes[0].set_ylabel("Probability density", fontsize=11)
    fig.suptitle(
        "Heston-Market Hedging Performance Across Transaction Costs",
        fontsize=17,
        fontweight="semibold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.015,
        (
            r"Neural networks are trained separately at each $\gamma$ using "
            r"$-\mathrm{CVaR}_{50\%}(G)$. Lower score is better."
        ),
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()

    for path in [output_png, output_pdf]:
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            kwargs = {"bbox_inches": "tight", "facecolor": "white"}
            if path.suffix.lower() == ".png":
                kwargs["dpi"] = 350
            fig.savefig(path, **kwargs)
    return fig, axes
