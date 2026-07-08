# Deep Hedging
Erdos Institute Summer 2026 Quant Finance Boot Camp Final Project

Group Members: Francis White, Ahmadreza Khazaeipoul, Jude Pereira, Antareep Gogoi, Gabriella Torres Nothaft 

---

The different files are:
- deep_hedging_replicating_figures.ipynb: Replicated figures 5 and 6 from Buehler et al, which focuses on understanding the conditional value at risk (CVaR) and the network structures (simple vs recurrent networks)
- deep-hedging-vs-black-scholes: Comparing the Neural Network hedge with the classical Black-Scholes Delta hedge, with and without transaction costs. 
- deep_hedging_markov folder: Using a 2-state Markov-switching volatility model to compare how simple and recurrent networks perform. Additionally, it includes an inclusion of a second hedging instrument to compare regime blind network versus a 2-instrument oracle.
- Delta_Vega_Deep_Hedging folder: Using a Delta-Vega hedging strategy to improve upon a Delta-only hedging strategt in the presence of transaction costs. More details to be found in the folder.
