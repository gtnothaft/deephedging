# Delta-Vega Deep Hedging

Deep hedging of a European call under the Heston stochastic-volatility model, comparing learned (LSTM + MLP) hedging strategies against a classical closed-form Greek-hedging benchmark, across multiple transaction-cost regimes.

## What this notebook does

1. **Simulates** asset-price and variance paths under Heston, and prices a target option `C1` (ATM call) and a vega-hedging instrument `C2` (OTM call) along each path with a semi-analytic Fourier pricer.
2. **Trains** recurrent neural-network hedgers under an entropic risk-measure loss:
   - `DeltaHedger` — trades only the underlying stock.
   - `DeltaVegaHedger` — trades the stock and `C2`.
   - Each is trained in both a **cost-aware** variant (transaction costs included in the training loss) and a **cost-ignorant** variant (trained as if trading were free, then evaluated under real costs), across two stock transaction-cost levels and (for delta-vega) two `C2` transaction-cost levels — 12 trained models in total.
3. **Builds an analytic benchmark**: a classical delta and delta-vega hedge constructed directly from Heston Greeks (via finite differences), with no cost-awareness mechanism, evaluated at the same cost regimes.
4. **Compares all strategies** — trained networks and the analytic benchmark — on premium-adjusted P&L (crediting each strategy the same Heston market-consistent premium for the option, so differences reflect hedging performance only, not pricing differences), reporting Mean / Std / VaR / CVaR and full P&L distributions, plus qualitative hedge-ratio trajectories along sample paths.

## Repo contents

```
delta_vega_dh.ipynb   the notebook
data/                 simulated path caches (*.pt, gitignored) and trained model checkpoints (*_best.pt, tracked)
```

The `data/*_best.pt` checkpoints for all 12 models are included in the repo, so the notebook will load them and skip training on a fresh clone. The larger simulated-path caches (`S_train.pt`, `C1_test.pt`, etc.) are gitignored and will be regenerated locally on first run (this is the slow step — see below).

## Requirements

- Python 3.9+
- `torch`, `numpy`, `pandas`, `matplotlib`, `tqdm`
- A CUDA GPU is strongly recommended — training simulates and trains on up to 1,000,000 paths per model. On CPU this will be slow; the notebook falls back to CPU automatically if no GPU is available.

```bash
pip install torch numpy pandas matplotlib tqdm jupyter
```

## Running it

Open `delta_vega_dh.ipynb` and run top to bottom.

- **Outside Colab**, data and checkpoints are cached to `./data/` (created automatically). On Colab, it mounts Google Drive and caches there instead.
- **First run**: path simulation (5 splits — train/val/price/test/plot) will take a while; results are cached to disk so subsequent runs skip this step.
- **Training**: each of the 12 models checks for an existing checkpoint in `data/` before training — since checkpoints are included in this repo, a fresh clone will load pretrained weights and skip straight to evaluation. Delete a `*_best.pt` file to force retraining that model.

## Key findings

See the notebook's **Conclusion** section for the full writeup with numbers. In brief:

- Vega hedging (adding `C2`) substantially reduces tail risk relative to delta-only hedging, for both the learned and analytic hedgers.
- Cost-aware training matters most when transaction costs are high — at low cost all approaches perform similarly, but at high cost cost-aware training wins decisively on both mean P&L and tail risk.
- The classical closed-form Greek hedge is not a free upper bound on hedging quality: it's the worst performer of all three approaches for delta-only hedging, and for delta-vega hedging it exhibits a real numerical fragility (its `C2` hedge ratio spikes as `C2`'s own vega collapses near maturity) that the learned hedgers avoid.
