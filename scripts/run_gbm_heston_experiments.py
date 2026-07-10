"""Run the complete GBM/Heston deep-hedging contribution.

This script produces:

* GBM tables/plots for ``gamma=0`` and ``gamma=0.005``;
* a Heston cost sweep for ``gamma in {0, 0.001, 0.005}``;
* the final nine-row Heston score table;
* a publication-quality three-panel Heston P&L density figure;
* trained Heston policy state dictionaries and loss histories.

Examples
--------
Quick smoke test::

    python scripts/run_gbm_heston_experiments.py --quick

Final run::

    python scripts/run_gbm_heston_experiments.py

Heston section only::

    python scripts/run_gbm_heston_experiments.py --skip-gbm
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

torch.set_num_threads(min(torch.get_num_threads(), 4))

from src.deep_hedging_gbm_heston import (
    ExperimentConfig,
    SimplePolicy,
    RecurrentPolicy,
    HestonSimplePolicy,
    HestonRecurrentPolicy,
    initial_bs_price,
    initial_heston_price,
    simulate_gbm,
    simulate_heston,
    ewma_variance_proxy,
    simulate_heston_with_observable_state,
    train_policy,
    pnl_cost_turnover_bs_delta,
    heston_delta_path,
    pnl_cost_turnover_heston_delta,
    pnl_cost_turnover_policy,
    comparison_table,
    cost_sweep_table,
)
from src.deep_hedging_gbm_heston.plots import (
    plot_pnl_distributions,
    plot_heston_cost_sweep_density,
)


HESTON_STRATEGIES = (
    "Analytical Heston delta",
    "Simple NN",
    "Recurrent NN",
)


def _gamma_tag(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def _seeded_policy(policy_class, cfg, seed_offset: int):
    torch.manual_seed(cfg.seed + seed_offset)
    return policy_class(cfg)


def run_gbm_scenario(
    cfg: ExperimentConfig,
    S_test: torch.Tensor,
    *,
    label: str,
    output_dir: Path,
):
    """Train GBM policies and compare them with analytical BS delta."""
    premium = initial_bs_price(cfg)
    simple = _seeded_policy(SimplePolicy, cfg, 101)
    recurrent = _seeded_policy(RecurrentPolicy, cfg, 202)

    print(f"\nTraining Simple NN: {label}")
    simple_history = train_policy(
        simple,
        cfg,
        simulate_gbm,
        name="Simple NN",
        initial_premium=premium,
    )
    print(f"\nTraining Recurrent NN: {label}")
    recurrent_history = train_policy(
        recurrent,
        cfg,
        simulate_gbm,
        name="Recurrent NN",
        initial_premium=premium,
    )

    with torch.no_grad():
        results = {
            "Black-Scholes delta": pnl_cost_turnover_bs_delta(
                S_test, cfg, premium
            ),
            "Simple NN": pnl_cost_turnover_policy(
                simple, S_test, cfg, premium
            ),
            "Recurrent NN": pnl_cost_turnover_policy(
                recurrent, S_test, cfg, premium
            ),
        }

    table = comparison_table(
        results, cvar_confidence=cfg.cvar_confidence
    ).round(6)
    tag = _gamma_tag(cfg.cost_gamma)
    table.to_csv(output_dir / "tables" / f"gbm_gamma_{tag}_metrics.csv")

    pd.DataFrame(
        {
            "epoch": range(1, len(simple_history) + 1),
            "Simple NN": simple_history,
            "Recurrent NN": recurrent_history,
        }
    ).to_csv(
        output_dir / "tables" / f"gbm_gamma_{tag}_training_loss.csv",
        index=False,
    )

    torch.save(
        simple.state_dict(),
        output_dir / "models" / f"gbm_simple_gamma_{tag}.pt",
    )
    torch.save(
        recurrent.state_dict(),
        output_dir / "models" / f"gbm_recurrent_gamma_{tag}.pt",
    )

    plot_pnl_distributions(
        {name: values[0] for name, values in results.items()},
        adjust_mean=False,
        title=rf"GBM market: terminal net P&L ($\gamma={cfg.cost_gamma:g}$)",
        save_path=output_dir / "figures" / f"gbm_gamma_{tag}_pnl.png",
    )

    return {
        "table": table,
        "results": results,
        "simple_history": simple_history,
        "recurrent_history": recurrent_history,
        "simple_policy": simple,
        "recurrent_policy": recurrent,
        "premium": premium,
    }


def run_heston_scenario(
    cfg: ExperimentConfig,
    test_sample: tuple[torch.Tensor, torch.Tensor],
    *,
    analytical_delta_path: torch.Tensor,
    label: str,
    output_dir: Path,
):
    """Train observable-state NNs and compare only three Heston strategies.

    The neural networks receive an EWMA variance proxy computed from observed
    stock returns. The analytical Heston delta receives the true simulated
    latent variance and is therefore labeled as an oracle model benchmark.
    """
    premium = initial_heston_price(cfg)
    simple = _seeded_policy(HestonSimplePolicy, cfg, 303)
    recurrent = _seeded_policy(HestonRecurrentPolicy, cfg, 404)

    print(f"\nTraining Simple NN: {label}")
    simple_history = train_policy(
        simple,
        cfg,
        simulate_heston_with_observable_state,
        name="Simple NN",
        initial_premium=premium,
    )
    print(f"\nTraining Recurrent NN: {label}")
    recurrent_history = train_policy(
        recurrent,
        cfg,
        simulate_heston_with_observable_state,
        name="Recurrent NN",
        initial_premium=premium,
    )

    S_test, v_true_test = test_sample
    v_hat_test = ewma_variance_proxy(S_test, cfg)

    with torch.no_grad():
        results = {
            "Analytical Heston delta": pnl_cost_turnover_heston_delta(
                S_test,
                v_true_test,
                cfg,
                premium,
                delta_path=analytical_delta_path,
            ),
            "Simple NN": pnl_cost_turnover_policy(
                simple,
                S_test,
                cfg,
                premium,
                extra_state=v_hat_test,
            ),
            "Recurrent NN": pnl_cost_turnover_policy(
                recurrent,
                S_test,
                cfg,
                premium,
                extra_state=v_hat_test,
            ),
        }

    assert tuple(results) == HESTON_STRATEGIES
    table = comparison_table(
        results, cvar_confidence=cfg.cvar_confidence
    ).round(6)
    tag = _gamma_tag(cfg.cost_gamma)
    table.to_csv(output_dir / "tables" / f"heston_gamma_{tag}_metrics.csv")

    pd.DataFrame(
        {
            "epoch": range(1, len(simple_history) + 1),
            "Simple NN": simple_history,
            "Recurrent NN": recurrent_history,
        }
    ).to_csv(
        output_dir / "tables" / f"heston_gamma_{tag}_training_loss.csv",
        index=False,
    )

    torch.save(
        simple.state_dict(),
        output_dir / "models" / f"heston_simple_gamma_{tag}.pt",
    )
    torch.save(
        recurrent.state_dict(),
        output_dir / "models" / f"heston_recurrent_gamma_{tag}.pt",
    )

    return {
        "table": table,
        "results": results,
        "simple_history": simple_history,
        "recurrent_history": recurrent_history,
        "simple_policy": simple,
        "recurrent_policy": recurrent,
        "premium": premium,
        "v_hat_test": v_hat_test,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small configuration for a smoke test.",
    )
    parser.add_argument(
        "--skip-gbm",
        action="store_true",
        help="Run only the final Heston cost sweep.",
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-paths", type=int, default=100_000)
    parser.add_argument(
        "--heston-cost-gammas",
        type=float,
        nargs=3,
        default=(0.0, 0.001, 0.005),
        metavar=("GAMMA0", "GAMMA1", "GAMMA2"),
    )
    parser.add_argument(
        "--heston-quad",
        type=int,
        default=64,
        help="Gauss-Laguerre nodes for the Heston oracle delta.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        base = ExperimentConfig(
            epochs=3,
            batch_size=256,
            eval_paths=1_000,
            seed=123,
        )
        n_quad = min(args.heston_quad, 32)
    else:
        base = ExperimentConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_paths=args.eval_paths,
            seed=123,
        )
        n_quad = args.heston_quad

    output_dir = ROOT / "outputs"
    for subdir in ["tables", "figures", "models"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Common test paths make comparisons paired and reproducible.
    torch.manual_seed(base.seed + 10_000)
    S_gbm_test = simulate_gbm(base.eval_paths, base)

    torch.manual_seed(base.seed + 20_000)
    S_heston_test, v_true_test = simulate_heston(base.eval_paths, base)

    if not args.skip_gbm:
        print("\n" + "=" * 76)
        print("GBM / BLACK-SCHOLES SECTION")
        print("=" * 76)
        for gamma in (0.0, 0.005):
            cfg = base.with_updates(cost_gamma=gamma)
            experiment = run_gbm_scenario(
                cfg,
                S_gbm_test,
                label=f"GBM gamma={gamma:g}",
                output_dir=output_dir,
            )
            print(f"\nGBM metrics, gamma={gamma:g}")
            print(experiment["table"])

    print("\n" + "=" * 76)
    print("HESTON COST SWEEP")
    print("=" * 76)
    print("Computing the analytical Heston delta path once for all cost levels...")
    analytical_delta_path = heston_delta_path(
        S_heston_test,
        v_true_test,
        base,
        n_quad=n_quad,
    )

    heston_experiments: dict[float, dict] = {}
    for gamma in args.heston_cost_gammas:
        gamma = float(gamma)
        cfg = base.with_updates(cost_gamma=gamma)
        print("\n" + "-" * 76)
        print(f"HESTON TRANSACTION COST gamma={gamma:g}")
        print("-" * 76)
        experiment = run_heston_scenario(
            cfg,
            (S_heston_test, v_true_test),
            analytical_delta_path=analytical_delta_path,
            label=f"Heston gamma={gamma:g}",
            output_dir=output_dir,
        )
        heston_experiments[gamma] = experiment
        print(experiment["table"])

    final_table = cost_sweep_table(
        heston_experiments,
        cvar_confidence=base.cvar_confidence,
    ).round(6)
    final_table.to_csv(
        output_dir / "tables" / "heston_cost_sweep_metrics.csv",
        index=False,
    )

    print("\nFINAL HESTON COST-SWEEP TABLE")
    print(final_table.to_string(index=False))

    fig, _ = plot_heston_cost_sweep_density(
        heston_experiments,
        confidence=base.cvar_confidence,
        output_png=output_dir / "figures" / "heston_pnl_density_cost_sweep.png",
        output_pdf=output_dir / "figures" / "heston_pnl_density_cost_sweep.pdf",
    )
    # Prevent the command-line process from holding the figure open.
    import matplotlib.pyplot as plt

    plt.close(fig)

    print("\nSaved final deliverables:")
    print(output_dir / "tables" / "heston_cost_sweep_metrics.csv")
    print(output_dir / "figures" / "heston_pnl_density_cost_sweep.png")
    print(output_dir / "figures" / "heston_pnl_density_cost_sweep.pdf")


if __name__ == "__main__":
    main()
