# Deep Hedging vs. Black-Scholes

A single-file PyTorch project that trains a neural-network hedging strategy ("Deep Hedging") and compares it against the classical, closed-form Black-Scholes Delta hedge — with and without proportional transaction costs.

## Overview

This project implements the **Deep Hedging** framework introduced by Buehler, Gonon, Teichmann & Wood (2019), applied to hedging a European call option under simulated Black-Scholes (Geometric Brownian Motion) market dynamics.

Instead of a single network shared across time, the option's 30-day life is split into 29 rebalancing dates, each handled by its own small independent network (`DayNet_0` ... `DayNet_28`). Each network observes two inputs — today's log stock price and yesterday's position — and outputs today's target hedge position. Millions of simulated price paths are run through all 29 networks in sequence, and gradient descent (via PyTorch autograd / backpropagation through time) tunes every network's weights to optimize the **CVaR (Conditional Value at Risk)** of the resulting profit-and-loss distribution.

No option-pricing formula is hard-coded — the network discovers hedging behavior purely from simulated outcomes, which is then benchmarked against the textbook Black-Scholes Delta hedge.

## Features

- **Monte Carlo market simulator** — simulates GBM price paths in log-space (pure NumPy)
- **Black-Scholes closed-form benchmark** — analytic price, Delta, and a naive Delta-hedge P&L simulation
- **Deep Hedging neural network** — 29 independent per-day feed-forward networks (2 → 16 → 16 → 1, ReLU), trained end-to-end with backprop through time
- **CVaR risk measure / loss function** — Rockafellar–Uryasev convex loss, jointly optimized with the network weights
- **Two scenarios** — frictionless market vs. 1% proportional transaction costs
- **Automatic GPU support** — uses CUDA if available, falls back to CPU
- **Quick-test mode** — a `QUICK_TEST` flag shrinks the dataset/epoch count so the full pipeline can be sanity-checked in seconds before committing to the full run
- **Results & plots** — training curves, P&L histograms, learned hedge-ratio vs. Black-Scholes Delta comparison, transaction-cost comparison, and a summary CSV table

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- Pandas

Install dependencies:

```bash
pip install torch numpy scipy matplotlib pandas
```

## Usage

Run the full experiment:

```bash
python Ahmadreza__deep_Hedging_and_Black_Scholes.py
```

Before running the full experiment, it's recommended to first set `QUICK_TEST = True` near the top of the script to confirm the pipeline runs end-to-end in well under a minute. Set it back to `False` for the full run.

### Dataset sizes (full run)

| Split      | Paths     |
|------------|-----------|
| Training   | 1,000,000 |
| Validation | 100,000   |
| Test       | 100,000   |

### Runtime note

Training 1,000,000 paths through 29 sequential per-day networks is computationally heavy. This runs quickly on a CUDA-capable GPU, but can take a long time (potentially hours) on a CPU-only machine, depending on hardware and the number of epochs.

## Configuration

Key parameters can be adjusted at the top of the script:

| Parameter                 | Description                                      | Default |
|----------------------------|---------------------------------------------------|---------|
| `N_EPOCHS`                 | Training epochs                                   | 20      |
| `BATCH_SIZE`                | Paths per gradient step                            | 8192    |
| `LEARNING_RATE`             | Adam optimizer step size                           | 1e-2    |
| `HIDDEN_WIDTH`              | Neurons per hidden layer                           | 16      |
| `ALPHA`                     | CVaR confidence level                              | 0.5     |
| `COST_RATE_SCENARIOS`       | Transaction cost rates to compare                  | 0%, 1%  |
| `S0`, `K`, `SIGMA`, `R`     | Market/option parameters (spot, strike, vol, rate) | 100, 100, 0.20, 0.0 |
| `N_DAYS`                    | Number of daily price observations                 | 30      |

## Output

Running the script creates an `outputs_full_project/` directory containing:

- `training_curves.png` — training/validation loss convergence
- `pnl_hist_no_cost.png` — hedged P&L distribution, no transaction costs
- `pnl_hist_with_cost.png` — hedged P&L distribution, with transaction costs
- `delta_comparison.png` — learned hedge ratio vs. Black-Scholes Delta
- `cost_comparison.png` — mean transaction cost, Deep Hedge vs. naive Delta hedge
- `results_table.csv` — summary table (price, mean P&L, std P&L per scenario/method)
- `model_no_cost.pt`, `model_with_cost.pt` — trained model weights

## Reference

Buehler, H., Gonon, L., Teichmann, J., Wood, B., Mohan, B., & Kochems, J. (2019). *Deep Hedging: Hedging Derivatives Under Generic Market Frictions Using Reinforcement Learning*. Swiss Finance Institute Research Paper No. 19-80.

## License

Add a license of your choice (e.g. MIT) here.
