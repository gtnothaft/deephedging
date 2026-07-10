"""Experiment configuration for the GBM/Heston deep-hedging study."""
from dataclasses import dataclass, replace
import torch


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the single-option hedging experiments.

    The default contract is an at-the-money European call with one month to
    maturity. We use 22 hedge intervals to mimic daily rebalancing over one
    trading month.

    Parameters
    ----------
    cost_gamma:
        Proportional transaction-cost rate. For example, ``0.005`` means each
        stock trade costs 0.5% of traded notional.
    cvar_confidence:
        Confidence parameter in the lower-tail CVaR gain objective. At 0.50,
        empirical CVaR is the average of the lowest 50% of terminal gains.
    ewma_decay:
        Persistence parameter for the observable EWMA variance proxy. The
        default 0.94 follows the daily RiskMetrics convention [RiskMetrics1996].
    """

    # Contract and market horizon.
    S0: float = 100.0
    K: float = 100.0
    sigma: float = 0.20
    mu: float = 0.0
    r: float = 0.0
    T: float = 30 / 365
    N: int = 22

    # Hedging objective and market friction.
    cost_gamma: float = 0.0
    cvar_confidence: float = 0.50

    # Observable volatility proxy used by the Heston neural networks.
    ewma_decay: float = 0.94
    variance_floor: float = 1e-8

    # Neural-network training.
    hidden: int = 15
    batch_size: int = 4096
    epochs: int = 250
    lr: float = 5e-3
    grad_clip: float = 10.0
    eval_paths: int = 100_000
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Heston parameters used in the project experiment.
    heston_kappa: float = 3.0
    heston_theta: float = 0.20**2
    heston_xi: float = 0.60
    heston_rho: float = -0.70

    def with_updates(self, **kwargs) -> "ExperimentConfig":
        """Return a copy with selected fields changed."""
        return replace(self, **kwargs)
