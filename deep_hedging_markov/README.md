# Deep Hedging under Markov-Switching Volatility

Replication and extension of [Buehler et al. (2019)](https://arxiv.org/abs/1802.03042) under a two-state Markov-switching volatility model. The hedging objective is CVaR(50%) with no transaction costs.

## Notebooks

### 1. `deep_hedging_section42.ipynb` — Black-Scholes baseline
Replicates Section 4.2 of the paper in a Black-Scholes market (constant σ = 0.20). Establishes the core framework: CVaR loss, the action-position distinction, and two network architectures — a Simple MLP (no position feedback) and a Recurrent MLP (feeds the current position back as input at each step). The recurrent network nearly matches the BS delta hedge; the simple network does not.

### 2. `deep_hedging_markov.ipynb` — Markov switching, single instrument
Extends to a market where volatility switches between σ_L = 0.20 and σ_H = 0.70 following a continuous-time Markov chain (α = 10/yr, β = 20/yr). The regime is never revealed to the network. Introduces occupation-time option pricing, precomputed lookup tables, and a Markov delta oracle benchmark. The recurrent network (score 2.22) nearly matches the oracle (score 2.18) despite not observing the regime.

### 3. `deep_hedging_markov_2asset.ipynb` — Markov switching, two instruments
Adds a second hedging instrument: a 60-day OTM call (K₂ = 105). The second option's market price serves as an implicit regime signal. A two-instrument oracle (score 0.90) cuts the hedging cost roughly in half relative to the stock-only case (score 2.18). The recurrent network achieves a score of 0.63, competitive with the oracle, by learning to infer the hidden regime from the second option's price.

## Requirements

```
numpy
torch
scipy
matplotlib
```

## Results summary

| Strategy | Score ↓ |
|---|---|
| 2-instrument oracle (stock + OTM call, regime known) | 0.90 |
| Recurrent network (stock + OTM call, regime hidden) | 0.63 |
| Markov Δ oracle (stock only, regime known) | 2.18 |
| Recurrent network (stock only, regime hidden) | 2.22 |

Score = −CVaR₅₀%(G), where G is the hedging P&L net of the model price. Lower is better.
