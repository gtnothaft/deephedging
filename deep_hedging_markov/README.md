# Deep Hedging under Markov-Switching Volatility

Replication and extension of [Buehler et al. (2019)](https://arxiv.org/abs/1802.03042) under a two-state Markov-switching volatility model, using CVaR(50%) as the hedging objective with no transaction costs.

## Notebooks

**`deep_hedging_section42.ipynb`** — Black-Scholes baseline

Starting point. Replicates Section 4.2 of the paper in a constant-volatility Black-Scholes market (σ = 0.20). Two network architectures are compared: a Simple MLP with no position feedback, and a Recurrent MLP that takes the current position as an additional input at each step. The recurrent network nearly matches the BS delta; the simple one doesn't come close.

**`deep_hedging_markov.ipynb`** — Markov switching, stock only

Moves to a market where volatility switches between σ_L = 0.20 and σ_H = 0.70 via a continuous-time Markov chain (α = 10/yr, β = 20/yr). The network never observes the current regime. Pricing uses occupation-time distributions and precomputed lookup tables. The benchmark is the Markov delta, which has full regime knowledge. The recurrent network scores 2.22 against the oracle's 2.18.

**`deep_hedging_markov_2asset.ipynb`** — Markov switching, two instruments

Adds a 60-day OTM call (K₂ = 105) as a second hedging instrument. The key idea is that the second option's price leaks information about the hidden regime, giving the network an implicit signal it can exploit. The two-instrument oracle scores 0.90 — roughly half the cost of the stock-only case — and the recurrent network achieves 0.62.

## Requirements

```
numpy
torch
scipy
matplotlib
```

## Results

| Strategy | Score ↓ |
|---|---|
| 2-instrument oracle (stock + OTM call, regime known) | 0.90 |
| Recurrent network (stock + OTM call, regime hidden) | 0.62 |
| Markov Δ oracle (stock only, regime known) | 2.18 |
| Recurrent network (stock only, regime hidden) | 2.22 |

Score = −CVaR₅₀%(G), where G is the hedging P&L net of the model price. Lower is better.
