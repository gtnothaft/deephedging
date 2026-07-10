# Deep Hedging
Erdos Institute Summer 2026 Quant Finance Boot Camp Final Project

Group Members: Francis White, Ahmadreza Khazaeipoul, Jude Pereira, Antareep Gogoi, Gabriella Torres Nothaft 

---

The different files are:
- deep_hedging_replicating_figures.ipynb: Replicated figures 5 and 6 from Buehler et al, which focuses on understanding the conditional value at risk (CVaR) and the network structures (simple vs recurrent networks)
- deep-hedging-vs-black-scholes: Comparing the Neural Network hedge with the classical Black-Scholes Delta hedge, with and without transaction costs. 
- deep_hedging_markov folder: Using a 2-state Markov-switching volatility model to compare how simple and recurrent networks perform. Additionally, it includes an inclusion of a second hedging instrument to compare regime blind network versus a 2-instrument oracle.
- Delta_Vega_Deep_Hedging folder: Using a Delta-Vega hedging strategy to improve upon a Delta-only hedging strategt in the presence of transaction costs. More details to be found in the folder.

- Delta-only Deep Hedging on Heston Market:

This notebook compares analytical delta hedges with Simple and Recurrent neural-network hedge policies for an at-the-money European call. The implementation follows the deep-hedging policy-search framework of Bühler et al. [Buehler2019].

The contribution proceeds from GBM / Black-Scholes validation to a Heston stochastic-volatility cost sweep:

- GBM: `c_s = 0` and `0.005`, benchmarked against analytical Black-Scholes delta [BlackScholes1973].
- Heston: `c_s = 0`, `0.001`, and `0.005`, benchmarked against analytical Heston stock delta [Heston1993].

For the Heston neural networks, the NNs receive an EWMA variance estimate constructed causally from observed stock returns [RiskMetrics1996]. The Recurrent NN additionally observes its previous hedge position.

The networks are trained and scored using

```math
\mathrm{Score}=-\mathrm{CVaR}_{50\%}(G),
```

where $G$ is raw terminal net gain after transaction costs. Lower scores indicate better lower-tail hedging performance.

The final Heston tables and presentation figure contain exactly three strategies:

1. Analytical Heston delta — oracle benchmark using simulated latent variance.
2. Simple NN.
3. Recurrent NN.

Main files:

```text
notebooks/03_gbm_heston_deep_hedging.ipynb
src/deep_hedging_gbm_heston/
scripts/run_gbm_heston_experiments.py
docs/gbm_heston_method_note.md
docs/references.md
```