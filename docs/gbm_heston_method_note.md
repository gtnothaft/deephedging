# Method note: GBM/Heston deep hedging comparison

## 1. Research question

Can a neural-network hedge improve lower-tail terminal hedging performance relative to a model-based delta hedge when volatility is stochastic and stock rebalancing is costly?

The neural policies are trained directly against a terminal risk objective, following the deep-hedging framework of Bühler et al. [Buehler2019, BuehlerHorvath2022].

## 2. Contract and hedge instrument

The claim is a short, at-the-money European call with payoff

```math
(S_T-K)^+.
```

The only hedge instrument is the underlying stock. The hedge is rebalanced on 22 discrete dates over a 30-day maturity.

## 3. Market models

### 3.1 GBM / Black-Scholes validation

The GBM simulator uses

```math
S_{t+\Delta t}
=
S_t\exp\left[
(\mu-\tfrac12\sigma^2)\Delta t
+\sigma\sqrt{\Delta t}Z_t
\right].
```

The analytical Black-Scholes call delta is the classical GBM benchmark [BlackScholes1973].

### 3.2 Heston stochastic volatility

The Heston simulator uses

```math
\frac{dS_t}{S_t}
=
\mu\,dt+\sqrt{v_t}\,dW_t^S,
```

```math
dv_t
=
\kappa(\theta-v_t)dt
+\xi\sqrt{v_t}\,dW_t^v,
\qquad
\mathrm{corr}(dW^S,dW^v)=\rho.
```

Paths are simulated by full-truncation Euler. The Heston dynamics and semi-analytical option representation follow Heston (1993) [Heston1993].

## 4. Analytical Heston delta benchmark

The Heston call price is evaluated from the characteristic-function representation

$$
C=S P_1-Ke^{-r\tau}P_2.
$$

Holding the current variance fixed, the stock delta is

```math
\Delta^{\mathrm{Heston}}
=
\frac{\partial C}{\partial S}
=P_1.
```

The code evaluates the Fourier integrals using Gauss-Laguerre quadrature. At each hedge date it uses the simulator's true latent variance $v_t$. Therefore, the analytical Heston delta is a **model-consistent oracle stock-delta benchmark** as opposed to a real implementable strategy.

Because only the stock is tradable, the Heston market remains incomplete: stock delta hedging does not directly span volatility risk.

## 5. Observable state for the Heston neural networks

The latent Heston variance is not directly quoted in the market and is not passed to the neural networks. The NNs instead receive a causal EWMA variance proxy based only on observed stock returns [RiskMetrics1996]:

```math
\hat v_0=\sigma_0^2,
```

```math
\hat v_t
=
\lambda_{\mathrm{EWMA}}\hat v_{t-1}
+
(1-\lambda_{\mathrm{EWMA}})
\frac{\log^2(S_t/S_{t-1})}{\Delta t},
\qquad
\lambda_{\mathrm{EWMA}}=0.94.
```

The state features are:

- Simple NN:
  ```math
  [\log(S_t/K),\;\tau_t/T,\;\sqrt{\hat v_t}/\sigma_0].
  ```
- Recurrent NN:
  ```math
  [\log(S_t/K),\;\tau_t/T,\;\sqrt{\hat v_t}/\sigma_0,\;\delta_{t-1}].
  ```

The previous hedge is included because trading costs depend on the trade size $|\delta_t-\delta_{t-1}|$, so current holdings are relevant to the next action [Buehler2019].

## 6. Terminal net gain and transaction costs

For a short call, terminal net gain is

```math
G
=
V_0-(S_T-K)^+
+
\sum_{t=0}^{N-1}
\delta_t(S_{t+1}-S_t)
-
C_T,
```

with

```math
C_T
=
\gamma
\sum_{t=0}^{N-1}
S_t|\delta_t-\delta_{t-1}|
+
\gamma S_T|\delta_{N-1}|.
```

The final term liquidates the remaining stock position at maturity. The cost model is a stock-only proportional specialization of the generic convex trading-cost framework in deep hedging [Buehler2019].

## 7. Training objective

For terminal gain $G$, define lower-tail CVaR by

```math
\operatorname{CVaR}_{\alpha}(G)
=
\sup_w
\left[
w
-
\frac{1}{1-\alpha}
\mathbb E[(w-G)^+]
\right].
```

The project uses $\alpha=0.50$. The neural network and auxiliary threshold $w$ are trained jointly by minimizing

```math
\mathcal L(\theta,w)
=
-w
+
\frac{1}{1-\alpha}
\mathbb E[(w-G_\theta)^+].
```

Thus,

```math
\boxed{
\mathcal L
=-\operatorname{CVaR}_{50\%}(G)
}.
```

Optimization uses Adam [KingmaBa2015].

## 8. Primary out-of-sample score

The reported score is

```math
\boxed{
\mathrm{Score}
=-\operatorname{CVaR}_{50\%}(G)
}.
```

At the 50% level, empirical CVaR is the mean of the lowest half of out-of-sample terminal gains. Since gain is better when larger, negating CVaR gives a loss-like metric:

> **Lower score means better lower-tail hedging performance.**

Scores are calculated from raw terminal net gains. No mean-centering or visual tail clipping enters the metric.

## 9. Heston cost sweep

The final Heston experiment uses

```math
\gamma\in\{0,0.001,0.005\}.
```

The Simple and Recurrent networks are trained separately at every $\gamma$, because the optimal balance between risk reduction and turnover changes with transaction costs.

All strategies and all cost levels are evaluated on the same held-out Heston paths. The analytical Heston delta positions are computed once and reused across cost levels; only the cost deduction changes.

The Heston output contains exactly:

1. Analytical Heston delta.
2. Simple NN.
3. Recurrent NN.

Black-Scholes delta is not included in any Heston table, plot, or calculation.

## 10. Final figure

The final presentation figure is a three-panel density plot of raw terminal net P&L:

- left: $\gamma=0$,
- middle: $\gamma=0.001$,
- right: $\gamma=0.005$.

Each panel shows the analytical Heston delta, Simple NN, and Recurrent NN. The legend reports the primary score for each curve. The plotting range may trim the outer 0.1% tails for visual readability, but scores always use the complete sample.

## 11. Interpretation limits

Results are conditional on:

- the chosen Heston parameters,
- the full-truncation Euler simulator,
- the stock-only hedge universe,
- the proportional-cost model,
- the EWMA state proxy,
- the neural architecture and training budget,
- the fixed random seed and held-out test sample.

## References

See [`references.md`](references.md) and [`references.bib`](references.bib).
