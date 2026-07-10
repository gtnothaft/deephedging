"""Utilities for Antareep's GBM/Heston deep-hedging experiments."""
from .config import ExperimentConfig
from .black_scholes import bs_call_price_delta, initial_bs_price
from .heston import (
    heston_call_price_delta_numpy,
    heston_delta_torch,
    initial_heston_price,
)
from .market_models import (
    simulate_gbm,
    simulate_heston,
    ewma_variance_proxy,
    simulate_heston_with_observable_state,
)
from .policies import (
    SimplePolicy,
    RecurrentPolicy,
    HestonSimplePolicy,
    HestonRecurrentPolicy,
)
from .pnl import (
    terminal_pnl_policy,
    pnl_cost_turnover_policy,
    pnl_cost_turnover_from_delta_path,
    bs_delta_path,
    pnl_cost_turnover_bs_delta,
    heston_delta_path,
    pnl_cost_turnover_heston_delta,
)
from .training import (
    negative_cvar_oce_loss,
    train_policy,
    train_policy_with_sampler,
)
from .metrics import (
    SCORE_COLUMN,
    empirical_cvar_gain,
    negative_cvar_score,
    risk_metrics,
    comparison_table,
    cost_sweep_table,
)

__all__ = [
    "ExperimentConfig",
    "bs_call_price_delta",
    "initial_bs_price",
    "heston_call_price_delta_numpy",
    "heston_delta_torch",
    "initial_heston_price",
    "simulate_gbm",
    "simulate_heston",
    "ewma_variance_proxy",
    "simulate_heston_with_observable_state",
    "SimplePolicy",
    "RecurrentPolicy",
    "HestonSimplePolicy",
    "HestonRecurrentPolicy",
    "terminal_pnl_policy",
    "pnl_cost_turnover_policy",
    "pnl_cost_turnover_from_delta_path",
    "bs_delta_path",
    "pnl_cost_turnover_bs_delta",
    "heston_delta_path",
    "pnl_cost_turnover_heston_delta",
    "negative_cvar_oce_loss",
    "train_policy",
    "train_policy_with_sampler",
    "SCORE_COLUMN",
    "empirical_cvar_gain",
    "negative_cvar_score",
    "risk_metrics",
    "comparison_table",
    "cost_sweep_table",
]
