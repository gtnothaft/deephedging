from __future__ import annotations

import numpy as np
import torch

from src.deep_hedging_gbm_heston import (
    ExperimentConfig,
    HestonSimplePolicy,
    HestonRecurrentPolicy,
    empirical_cvar_gain,
    negative_cvar_score,
    ewma_variance_proxy,
    simulate_heston,
    heston_call_price_delta_numpy,
)
from src.deep_hedging_gbm_heston.plots import plot_heston_cost_sweep_density


def test_score_convention_is_negative_lower_half_mean():
    gain = np.array([-4.0, -2.0, 1.0, 3.0])
    assert np.isclose(empirical_cvar_gain(gain, confidence=0.50), -3.0)
    assert np.isclose(negative_cvar_score(gain, confidence=0.50), 3.0)


def test_ewma_variance_proxy_is_causal():
    cfg = ExperimentConfig(N=22, eval_paths=64, device="cpu")
    torch.manual_seed(7)
    S, _ = simulate_heston(64, cfg)

    S_changed = S.clone()
    S_changed[:, 10:] *= 1.20

    original = ewma_variance_proxy(S, cfg)
    changed = ewma_variance_proxy(S_changed, cfg)

    # Estimates through t=9 cannot depend on prices from t=10 onward.
    assert torch.allclose(original[:, :10], changed[:, :10])
    assert torch.isfinite(original).all()
    assert torch.all(original > 0)


def test_heston_nn_policies_accept_observable_state():
    cfg = ExperimentConfig(N=3, hidden=4, device="cpu")
    S_t = torch.tensor([90.0, 100.0, 110.0])
    v_hat_t = torch.tensor([0.03, 0.04, 0.05])
    previous = torch.tensor([0.2, 0.5, 0.8])

    simple = HestonSimplePolicy(cfg)
    recurrent = HestonRecurrentPolicy(cfg)

    simple_delta = simple.forward_one_step(
        0, S_t, previous_delta=previous, extra_state_t=v_hat_t
    )
    recurrent_delta = recurrent.forward_one_step(
        0, S_t, previous_delta=previous, extra_state_t=v_hat_t
    )

    assert simple_delta.shape == S_t.shape
    assert recurrent_delta.shape == S_t.shape
    assert torch.all((0 <= simple_delta) & (simple_delta <= 1))
    assert torch.all((0 <= recurrent_delta) & (recurrent_delta <= 1))


def test_heston_price_and_delta_are_finite_and_sensible():
    price, delta = heston_call_price_delta_numpy(
        S=100.0,
        v=0.20**2,
        K=100.0,
        tau=30 / 365,
        r=0.0,
        kappa=3.0,
        theta=0.20**2,
        xi=0.60,
        rho=-0.70,
        n_quad=32,
    )

    price = float(np.asarray(price))
    delta = float(np.asarray(delta))
    assert np.isfinite(price)
    assert np.isfinite(delta)
    assert 0.0 < price < 20.0
    assert 0.0 < delta < 1.0


def test_final_plot_uses_exactly_three_heston_strategies(tmp_path):
    rng = np.random.default_rng(123)
    strategies = (
        "Analytical Heston delta",
        "Simple NN",
        "Recurrent NN",
    )
    experiments = {}
    for gamma in (0.0, 0.001, 0.005):
        results = {}
        for i, strategy in enumerate(strategies):
            gain = torch.tensor(
                rng.normal(loc=-2 * gamma * 100, scale=1.0 + 0.1 * i, size=400),
                dtype=torch.float32,
            )
            zeros = torch.zeros_like(gain)
            results[strategy] = (gain, zeros, zeros)
        experiments[gamma] = {"results": results}

    png = tmp_path / "figure.png"
    pdf = tmp_path / "figure.pdf"
    fig, axes = plot_heston_cost_sweep_density(
        experiments,
        output_png=png,
        output_pdf=pdf,
    )

    assert len(axes) == 3
    assert png.exists() and png.stat().st_size > 0
    assert pdf.exists() and pdf.stat().st_size > 0

    import matplotlib.pyplot as plt

    plt.close(fig)
