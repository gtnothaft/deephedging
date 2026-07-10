# Deep hedging under GBM and Heston markets

This folder contains Antareep Gogoi's contribution to the Erdős Institute Quant Finance project. It compares analytical delta hedges with neural-network hedge policies for a short at-the-money European call. The implementation follows the direct policy-search / deep-hedging framework of Bühler et al. [Buehler2019, BuehlerHorvath2022].

## Final experiment design

### Part I: GBM / Black-Scholes validation

Two scenarios are retained as sanity checks:

1. GBM with `cost_gamma = 0`.
2. GBM with `cost_gamma = 0.005`.

Strategies:

- analytical Black-Scholes delta [BlackScholes1973],
- Simple NN,
- Recurrent NN.

### Part II: Heston stochastic-volatility cost sweep

The final Heston analysis uses:

```text
cost_gamma = 0
cost_gamma = 0.001
cost_gamma = 0.005
```

At each transaction-cost level, the neural networks are trained **from scratch** against the corresponding cost model.

The Heston tables and figures compare exactly three strategies:

1. **Analytical Heston delta** — model-consistent stock delta computed from the semi-analytical Heston characteristic-function formula [Heston1993]. It uses the simulator's true latent variance and is therefore an **oracle benchmark**.
2. **Simple NN** — uses current log-moneyness, time to maturity, and an observable EWMA volatility estimate.
3. **Recurrent NN** — uses the same observable features plus the previous hedge position.

## Observable Heston state for the neural networks

The simulator's variance process `v_t` is latent and is not passed to the neural networks. Instead, the NNs use the causal EWMA estimate [RiskMetrics1996]

```math
\hat v_t
=
\lambda\hat v_{t-1}
+
(1-\lambda)
\frac{\log^2(S_t/S_{t-1})}{\Delta t},
\qquad \lambda=0.94.
```

The policy inputs are:

- Simple NN: `[log(S_t/K), tau/T, sqrt(v_hat_t)/sigma_0]`
- Recurrent NN: `[log(S_t/K), tau/T, sqrt(v_hat_t)/sigma_0, previous_delta]`

The analytical Heston delta still uses the true simulated `v_t`, which is why it must be labeled as an oracle model benchmark in the notebook, slide, and report.

## Training objective and score

The networks are trained by minimizing

```math
\boxed{
\mathrm{Loss}
=
-\mathrm{CVaR}_{50\%}(G)
}
```

where `G` is terminal net hedging gain after option payoff and transaction costs. The same quantity is the primary out-of-sample score:

```math
\boxed{
\mathrm{Score}
=
-\mathrm{CVaR}_{50\%}(G)
}
```

**Lower score is better.**

At the 50% confidence level, empirical CVaR is the average of the lowest half of terminal gains. The score is calculated from raw terminal net P&L; it is not mean-centered.

## Main files

```text
notebooks/
  03_gbm_heston_deep_hedging.ipynb

src/deep_hedging_gbm_heston/
  config.py          # experiment parameters
  black_scholes.py   # BS price and delta
  heston.py          # semi-analytical Heston price and delta
  market_models.py   # GBM/Heston simulation and EWMA variance proxy
  policies.py        # Simple/Recurrent GBM and Heston policies
  pnl.py             # terminal gain, cost, turnover, delta paths
  training.py        # -CVaR50% training objective
  metrics.py         # score and comparison tables
  plots.py           # final Heston cost-sweep density figure

scripts/
  run_gbm_heston_experiments.py

tests/
  test_gbm_heston_patch.py

docs/
  gbm_heston_method_note.md
  references.md
  references.bib
  repo_integration_checklist.md
```

## Quick validation

From the repository root:

```bash
python -m pip install -r requirements-antareep.txt
pytest -q tests/test_gbm_heston_patch.py
python scripts/run_gbm_heston_experiments.py --quick
```

## Final run

From the repository root:

```bash
python scripts/run_gbm_heston_experiments.py
```

The final run writes:

```text
outputs/tables/heston_cost_sweep_metrics.csv
outputs/figures/heston_pnl_density_cost_sweep.png
outputs/figures/heston_pnl_density_cost_sweep.pdf
outputs/models/heston_simple_gamma_*.pt
outputs/models/heston_recurrent_gamma_*.pt
```

## References

See [`docs/references.md`](docs/references.md) and [`docs/references.bib`](docs/references.bib).
