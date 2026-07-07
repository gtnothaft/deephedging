"""
================================================================================
 DEEP HEDGING vs. BLACK-SCHOLES  --  one-file PyTorch project
================================================================================

WHAT THIS FILE DOES
--------------------
Trains a neural-network hedging strategy ("Deep Hedging", Buehler, Gonon,
Teichmann & Wood, 2019) to hedge a European call option under simulated
Black-Scholes market dynamics, and compares it against the classical,
closed-form Black-Scholes Delta hedge -- with and without proportional
transaction costs.

THE BIG PICTURE (read this before the code)
---------------------------------------------
We hedge an option over 30 trading days (= 29 rebalancing decisions).
Instead of ONE neural network shared across all 29 days, we use 29
*independent* tiny networks, one per day -- "DayNet_0" decides the position
on day 0, "DayNet_1" decides the position on day 1, etc. Each network sees
two numbers: today's (log) stock price, and yesterday's position. We
simulate millions of fake price paths, run all 29 networks in sequence
along each path, total up the trading profit/loss and transaction costs,
and then use gradient descent (PyTorch autograd) to adjust all 29
networks' weights so that the *risk-adjusted* profit/loss distribution
(measured by CVaR -- Conditional Value at Risk) is as good as possible.
No option-pricing formula is hard-coded anywhere -- the network has to
*discover* good hedging behaviour purely from simulated outcomes.

DATA SIZES (as requested)
--------------------------
    Training paths    : 1,000,000
    Validation paths   :  100,000
    Test paths         :  100,000

A NOTE ON RUNTIME
------------------
1,000,000 training paths x 29 sequential per-day networks is a LOT of
computation per epoch. On a GPU this is quite fast (the 29 small networks
+ large batches are exactly what GPUs are good at). On a CPU-only machine
this can take a long time (potentially hours, depending on your hardware
and how many epochs you run). Two pieces of practical advice baked into
this file:
    1. Flip `QUICK_TEST = True` below to instantly shrink the dataset and
       epoch count, so you can confirm the whole pipeline runs correctly
       in well under a minute before committing to the full 1M run.
    2. If you have a CUDA GPU, this script will automatically use it
       (see `DEVICE` below) -- no code changes needed.

HOW TO RUN
----------
    pip install torch numpy scipy matplotlib pandas
    python deep_hedging_full_project.py

Reference: Buehler, Gonon, Teichmann, Wood, Mohan & Kochems (2019),
"Deep Hedging: Hedging Derivatives Under Generic Market Frictions Using
Reinforcement Learning," Swiss Finance Institute Research Paper No. 19-80.
================================================================================
"""

# =============================================================================
# SECTION 0: IMPORTS
# =============================================================================
import os                      # for creating the output folder
import time                    # for timing training
import dataclasses             # a clean way to hold configuration values

import numpy as np              # used for data simulation (no learning here)
from scipy.stats import norm    # the Normal-distribution CDF, for Black-Scholes

import torch                    # the deep learning library
import torch.nn as nn           # neural-network building blocks (layers, etc.)

import matplotlib
matplotlib.use("Agg")           # "Agg" = draw to PNG files, no on-screen window needed
import matplotlib.pyplot as plt
import pandas as pd             # for the final results table


# =============================================================================
# SECTION 1: USER SETTINGS  -- the knobs you are most likely to want to change
# =============================================================================

# --- Quick-test switch ------------------------------------------------------
# Set this to True the FIRST time you run the file, just to make sure
# everything works end-to-end in a few seconds, before launching the full
# 1-million-path training run (which can take a long time on a CPU).
QUICK_TEST = False

# --- Dataset sizes (as requested: 1M / 100k / 100k) -------------------------
N_TRAIN = 1_000_000
N_VAL = 100_000
N_TEST = 100_000

if QUICK_TEST:
    # Shrink everything drastically so the whole script finishes almost
    # instantly -- useful for debugging / sanity-checking on your laptop.
    N_TRAIN, N_VAL, N_TEST = 5_000, 1_000, 1_000

# --- Training hyperparameters ------------------------------------------------
N_EPOCHS = 3 if QUICK_TEST else 20    # how many full passes over the training data
BATCH_SIZE = 1024 if QUICK_TEST else 8192   # number of paths per gradient step
LEARNING_RATE = 1e-2                  # Adam optimizer step size
HIDDEN_WIDTH = 16                     # neurons per hidden layer in each per-day network
ALPHA = 0.5                           # CVaR confidence level (0.5 = mildly risk-averse)

# --- The two scenarios we compare: with and without transaction costs ------
COST_RATE_SCENARIOS = {
    "no_cost": 0.0,      # frictionless market
    "with_cost": 0.01,   # 1% proportional transaction cost
}

# --- Market / option parameters ---------------------------------------------
S0 = 100.0      # today's stock price
K = 100.0       # option strike (at-the-money call)
SIGMA = 0.20    # annualized volatility (20%)
R = 0.0         # risk-free rate (kept at zero, as in the reference paper)
N_DAYS = 30     # 30 daily price observations -> 29 rebalancing dates -> 29 networks

# --- Misc ---------------------------------------------------------------
DAY_IDX_FOR_DELTA_PLOT = 15   # which of the 29 days to use for the "hedge ratio" plot
RANDOM_SEED = 42
OUTPUT_DIR = "outputs_full_project"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Automatically use a GPU if one is available, otherwise fall back to CPU.
# Every tensor we want to compute with the network must live on this device.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# SECTION 2: MARKET CONFIGURATION & PRICE-PATH SIMULATOR  (pure NumPy)
# =============================================================================
# Nothing in this section is "learned" -- it is just a random-number-driven
# data generator that stands in for "the real market". The neural network
# never sees the formulas below; it only ever sees the simulated numbers
# they produce.

@dataclasses.dataclass
class MarketConfig:
    """Bundles all the market/option constants together so we can pass one
    object around instead of five separate numbers."""
    S0: float
    K: float
    sigma: float
    r: float
    n_days: int
    days_per_year: int = 365

    @property
    def n_steps(self) -> int:
        """Number of hedging decisions = number of per-day neural networks.
        n_days price points (S_0, ..., S_{n_days-1}) give n_days-1 gaps
        between them; the very last point is maturity itself (settlement,
        no further trading decision needed)."""
        return self.n_days - 1

    @property
    def T(self) -> float:
        """Total option life expressed as a fraction of a year."""
        return (self.n_days - 1) / self.days_per_year

    @property
    def dt(self) -> float:
        """Length of one time step, in years."""
        return self.T / self.n_steps

    @property
    def tau_grid(self) -> np.ndarray:
        """Time-to-maturity (in years) at each of the n_steps decision dates
        (used only by the Black-Scholes benchmark, which needs to know how
        much time is left at every rebalancing date)."""
        t = np.arange(self.n_steps) * self.dt
        return self.T - t

def simulate_gbm_paths(cfg: MarketConfig, n_paths: int, seed: int = None) -> np.ndarray:
    """Simulate `n_paths` independent Geometric Brownian Motion (Black-Scholes)
    price paths:                d S / S = (r - 0.5*sigma^2) dt + sigma dW

    Returns
    -------
    S : ndarray, shape (n_paths, cfg.n_days)
        S[:, 0]  = cfg.S0 for every path (today's known price)
        S[:, -1] = the simulated terminal price S_T
    """
    rng = np.random.default_rng(seed)                       # reproducible randomness
    dt = cfg.dt

    # One independent standard-normal random shock per path, per time step.
    z = rng.standard_normal((n_paths, cfg.n_steps))

    # Working in log-price space avoids prices ever going negative, and the
    # "-0.5*sigma^2*dt" term is the Ito correction that arises from that
    # change of variables (it is NOT a typo / NOT the same as the drift r).
    drift = (cfg.r - 0.5 * cfg.sigma ** 2) * dt
    diffusion = cfg.sigma * np.sqrt(dt) * z                  # Brownian increments scale with sqrt(time)
    log_increments = drift + diffusion                       # shape (n_paths, n_steps)

    # Cumulative sum turns "change at each step" into "total change so far",
    # i.e. it builds the random walk.
    log_paths = np.cumsum(log_increments, axis=1)

    # Prepend a column of zeros for "no change yet" at t=0.
    log_paths = np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1)

    # Exponentiate back from log-price to price.
    return cfg.S0 * np.exp(log_paths)


def make_datasets(cfg: MarketConfig, n_train: int, n_val: int, n_test: int, seed: int):
    """Generate three *disjoint* sets of simulated paths (different random
    seeds), so the model is trained, tuned, and finally scored on data it
    has never touched before -- standard machine-learning hygiene."""
    S_train = simulate_gbm_paths(cfg, n_train, seed=seed)
    S_val = simulate_gbm_paths(cfg, n_val, seed=seed + 1)
    S_test = simulate_gbm_paths(cfg, n_test, seed=seed + 2)
    return S_train, S_val, S_test


# =============================================================================
# SECTION 3: BLACK-SCHOLES CLOSED-FORM BENCHMARK  (pure NumPy)
# =============================================================================
# This is the classical textbook hedge we compare the neural network
# against. It is computed from a formula (no learning involved).

def bs_d1_d2(S, K, tau, sigma, r=0.0):
    """The two standard Black-Scholes intermediate quantities d1 and d2."""
    S = np.asarray(S, dtype=float)
    tau_safe = np.clip(np.asarray(tau, dtype=float), 1e-12, None)   # avoid div-by-zero at maturity
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau_safe) / (sigma * np.sqrt(tau_safe))
    d2 = d1 - sigma * np.sqrt(tau_safe)
    return d1, d2


def bs_call_price(S, K, tau, sigma, r=0.0):
    """Closed-form Black-Scholes price of a European call option."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    intrinsic = np.maximum(S - K, 0.0)                      # value if tau were exactly 0
    d1, d2 = bs_d1_d2(S, K, tau, sigma, r)
    price = S * norm.cdf(d1) - K * np.exp(-r * np.clip(tau, 0, None)) * norm.cdf(d2)
    return np.where(tau > 1e-12, price, intrinsic)


def bs_call_delta(S, K, tau, sigma, r=0.0):
    """Closed-form Black-Scholes Delta (the textbook hedge ratio) of a
    European call: how many shares of stock to hold per option sold."""
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d1, _ = bs_d1_d2(S, K, tau, sigma, r)
    delta_continuous = norm.cdf(d1)
    delta_at_maturity = (S > K).astype(float)                # the limiting 0/1 step function
    return np.where(tau > 1e-12, delta_continuous, delta_at_maturity)


def bs_delta_hedge_pnl(S_paths, K, sigma, r, tau_grid, cost_rate=0.0):
    """Simulate the classical strategy of rebalancing to the closed-form
    Black-Scholes Delta every day, and return the resulting profit/loss
    decomposition (trading P&L and transaction costs paid).
    """
    n_paths, n_points = S_paths.shape
    n_steps = n_points - 1
    deltas = np.zeros((n_paths, n_steps))
    delta_prev = np.zeros(n_paths)
    hedge_pnl = np.zeros(n_paths)
    total_cost = np.zeros(n_paths)

    for t in range(n_steps):
        delta_t = bs_call_delta(S_paths[:, t], K, tau_grid[t], sigma, r)
        trade = delta_t - delta_prev
        total_cost += cost_rate * S_paths[:, t] * np.abs(trade)
        hedge_pnl += delta_t * (S_paths[:, t + 1] - S_paths[:, t])
        deltas[:, t] = delta_t
        delta_prev = delta_t

    return hedge_pnl, total_cost, deltas


# =============================================================================
# SECTION 4: THE NEURAL NETWORK -- the hedging policy itself (PyTorch)
# =============================================================================

class StepNetwork(nn.Module):
    """The tiny feed-forward network used to make ONE day's hedging
    decision.  Architecture:   2 -> 16 -> 16 -> 1   with ReLU activations.

    Inputs  (2 numbers): [ log(S_t / S_0),  position held since yesterday ]
    Output  (1 number) : target position to hold today (shares of stock)
    """

    def __init__(self, n_in: int = 2, n_hidden: int = HIDDEN_WIDTH):
        super().__init__()                     # mandatory nn.Module setup
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),          # layer 1: 2 inputs -> n_hidden
            nn.ReLU(),                          # nonlinearity (lets the net learn curves, not just lines)
            nn.Linear(n_hidden, n_hidden),      # layer 2: n_hidden -> n_hidden
            nn.ReLU(),
            nn.Linear(n_hidden, 1),             # output layer: n_hidden -> 1 (no activation: any real number)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (batch, 2); self.net(x) has shape (batch, 1);
        # .squeeze(-1) drops that trailing size-1 dimension -> shape (batch,)
        return self.net(x).squeeze(-1)


class DeepHedger(nn.Module):
    """The full multi-day hedging strategy: `n_steps` independent
    StepNetworks (one per trading day) applied in sequence along a batch
    of simulated price paths, keeping a running tally of trading P&L and
    transaction costs.
    """
    def __init__(self, n_steps: int, n_hidden: int = HIDDEN_WIDTH):
        super().__init__()
        self.n_steps = n_steps
        # nn.ModuleList (NOT a plain Python list!) registers every network
        # inside it so PyTorch's autograd/optimizer can see and update
        # ALL 29 sets of weights automatically.
        self.day_networks = nn.ModuleList(
            [StepNetwork(n_in=2, n_hidden=n_hidden) for _ in range(n_steps)]
        )
    

    def forward(self, S: torch.Tensor, cost_rate: float = 0.0):
        """
        Parameters
        ----------
        S : torch.Tensor, shape (batch, n_steps + 1)
            Simulated price paths already moved onto the correct device.
        cost_rate : float
            Proportional transaction-cost rate (e.g. 0.01 = 1%).

        Returns
        -------
        hedge_pnl  : (batch,) total trading profit/loss over the option's life
        total_cost : (batch,) total transaction costs paid
        deltas     : (batch, n_steps) the position held over each interval
                      (kept around only so we can plot it later)
        """
        batch = S.shape[0]
    
        # log(S_t / S_0): a scale-free, zero-centered feature -- much easier
        # for a neural network to learn from than the raw price level.
        logS = torch.log(S / S[:, :1])

        # Running state, all shape (batch,), all initialised to zero.
        delta_prev = torch.zeros(batch, device=S.device)   # position held "yesterday" (none, before day 0)
        hedge_pnl = torch.zeros(batch, device=S.device)    # running trading P&L
        total_cost = torch.zeros(batch, device=S.device)   # running transaction costs
        deltas = []                                         # we'll collect each day's position here

        # Walk through all 29 trading days, one network per day.
        # Because every operation here is a tensor op, PyTorch automatically
        # builds a computation graph across all 29 steps, so gradients can
        # later flow backward through the entire sequence (this is exactly
        # "backpropagation through time").
        for t in range(self.n_steps):
            # Build today's 2-number input: [today's log-return, yesterday's position]
            feat = torch.stack([logS[:, t], delta_prev], dim=1)     # shape (batch, 2)

            # Ask DAY t's dedicated network for today's target position.
            delta_t = self.day_networks[t](feat)                    # shape (batch,)

            # How many shares we buy(+)/sell(-) to move from delta_prev to delta_t.
            trade_t = delta_t - delta_prev

            # Proportional transaction cost: rate * price * |shares traded|.
            cost_t = cost_rate * S[:, t] * trade_t.abs()

            # Trading P&L earned holding delta_t shares from today to tomorrow.
            hedge_pnl = hedge_pnl + delta_t * (S[:, t + 1] - S[:, t])
            total_cost = total_cost + cost_t

            deltas.append(delta_t)
            delta_prev = delta_t   # today's position becomes "yesterday's" for the next loop iteration

        deltas = torch.stack(deltas, dim=1)    # shape (batch, n_steps)
        return hedge_pnl, total_cost, deltas


# =============================================================================
# SECTION 5: RISK MEASURE / LOSS FUNCTION  (PyTorch) -- CVaR via the
# Rockafellar-Uryasev "Optimized Certainty Equivalent" trick
# =============================================================================
# We don't have "correct labels" to fit -- there's no such thing as the one
# right hedge ratio. Instead we score an entire DISTRIBUTION of simulated
# outcomes by how good its worst-case tail looks (this is what makes the
# objective "risk-adjusted" rather than just "average").

class CVaRLoss(nn.Module):
    """Convex risk-measure loss.  Minimizing this loss (jointly over the
    DeepHedger's weights AND the extra learnable scalar `w` defined below)
    is mathematically equivalent to maximizing the CVaR_alpha risk-adjusted
    return of the profit/loss distribution X. At the optimum, the loss
    value itself equals the risk-adjusted price the position requires.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        assert 0.0 < alpha < 1.0
        self.alpha = alpha
        self.lam = 1.0 / (1.0 - alpha)     # tail-loss weighting (e.g. alpha=0.99 -> lam=100)
        # `w` is an auxiliary threshold, learned jointly with the network by
        # gradient descent -- this is what avoids having to differentiate
        # through an explicit (and awkward) quantile/sorting operation.
        self.w = nn.Parameter(torch.tensor(0.0))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X = realized profit/loss per path (positive = profit).
        # -X = the loss. "shortfall" = how far a loss exceeds threshold w
        # (and exactly 0 for paths whose loss didn't exceed w).
        shortfall = torch.relu(-X - self.w)
        return self.w + self.lam * shortfall.mean()


# =============================================================================
# SECTION 6: TRAINING LOOP
# =============================================================================

def train_deep_hedge(S_train: np.ndarray, payoff_train: np.ndarray,
                      S_val: np.ndarray, payoff_val: np.ndarray, *,
                      n_steps: int, cost_rate: float, alpha: float,
                      n_hidden: int, lr: float, batch_size: int, n_epochs: int,
                      device: torch.device, seed: int = 0, verbose: bool = True):
    """Train one DeepHedger + CVaRLoss pair and return them, plus a record
    of the training/validation loss over time."""
    torch.manual_seed(seed)   # reproducible weight initialization

    model = DeepHedger(n_steps=n_steps, n_hidden=n_hidden).to(device)
    loss_fn = CVaRLoss(alpha=alpha).to(device)

    # ONE optimizer updates BOTH the 29 networks' weights and the CVaR
    # threshold `w` together, with a single .backward()/.step() per batch.
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()), lr=lr
    )

    # Move the full datasets onto the training device ONCE. With 1,000,000
    # paths x 30 floats x 4 bytes (float32) this is only ~120MB, so it
    # comfortably fits in GPU memory (or RAM) all at once -- this lets us
    # avoid the overhead of a DataLoader and just index the tensor directly.
    S_train_t = torch.tensor(S_train, dtype=torch.float32, device=device)
    payoff_train_t = torch.tensor(payoff_train, dtype=torch.float32, device=device)
    S_val_t = torch.tensor(S_val, dtype=torch.float32, device=device)
    payoff_val_t = torch.tensor(payoff_val, dtype=torch.float32, device=device)

    n_train = S_train_t.shape[0]
    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()                                    # enable training mode (relevant if using e.g. dropout)
        perm = torch.randperm(n_train, device=device)     # freshly re-shuffle every epoch
        epoch_losses = []

        # Manual mini-batching: walk through the shuffled indices in chunks
        # of `batch_size`. This is "mini-batch gradient descent": we update
        # the weights many times per epoch using a subset of the data each
        # time, instead of once per epoch using everything.
        for i in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            S_batch = S_train_t[idx]
            payoff_batch = payoff_train_t[idx]

            # ---- forward pass ----
            hedge_pnl, total_cost, _ = model(S_batch, cost_rate=cost_rate)
            # Profit/loss of: (short the option) + (hedge trading) - (costs).
            # No price is charged here (p0 = 0); by cash-invariance of the
            # risk measure, the converged loss value below IS the
            # risk-adjusted price -- we don't need to add anything.
            X = -payoff_batch + hedge_pnl - total_cost

            loss = loss_fn(X)

            # ---- backward pass + weight update ----
            optimizer.zero_grad()   # clear old gradients (PyTorch accumulates by default)
            loss.backward()        # autograd computes d(loss)/d(every parameter)
            optimizer.step()       # Adam nudges every parameter to reduce the loss

            epoch_losses.append(loss.item())

        # Periodically check validation-set performance (full batch, no
        # gradient tracking needed -> wrap in torch.no_grad() to save memory).
        if epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                hedge_pnl_v, cost_v, _ = model(S_val_t, cost_rate=cost_rate)
                X_v = -payoff_val_t + hedge_pnl_v - cost_v
                val_loss = loss_fn(X_v).item()
            train_loss = float(np.mean(epoch_losses))
            history["train_loss"].append((epoch, train_loss))
            history["val_loss"].append((epoch, val_loss))
            if verbose:
                print(f"  epoch {epoch:4d}/{n_epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"w={loss_fn.w.item():.4f}  ({time.time() - t0:.1f}s elapsed)")

    return model, loss_fn, history


# =============================================================================
# SECTION 7: EVALUATION ?
# =============================================================================

@torch.no_grad()           # disables gradient tracking for everything inside -> faster, less memory
def evaluate(model: DeepHedger, loss_fn: CVaRLoss, S: np.ndarray, payoff: np.ndarray,
             cost_rate: float, device: torch.device):
    """Run a trained model on held-out data and compute its risk-adjusted
    price exactly (via the closed-form CVaR quantile formula, rather than
    just trusting the learned `w`, for a clean, optimizer-independent score).
    """
    model.eval()
    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    payoff_t = torch.tensor(payoff, dtype=torch.float32, device=device)

    hedge_pnl, total_cost, deltas = model(S_t, cost_rate=cost_rate)
    X = -payoff_t + hedge_pnl - total_cost

    X_np = X.cpu().numpy()
    deltas_np = deltas.cpu().numpy()
    cost_np = total_cost.cpu().numpy()

    # Closed-form CVaR: the Rockafellar-Uryasev minimizer over w is exactly
    # the alpha-quantile of the loss, so we can compute CVaR exactly here
    # (no optimization needed) for a trustworthy, exact evaluation score.
    L = -X_np
    w_star = np.quantile(L, loss_fn.alpha)
    price = w_star + loss_fn.lam * np.maximum(L - w_star, 0).mean()

    return dict(X=X_np, deltas=deltas_np, cost=cost_np, price=price,
                mean_pnl=float(X_np.mean()), std_pnl=float(X_np.std()))


def evaluate_bs_delta_hedge(S: np.ndarray, payoff: np.ndarray, cfg: MarketConfig,
                             cost_rate: float, alpha: float):
    """Same evaluation, but for the classical Black-Scholes Delta hedge
    (no neural network -- pure closed-form formula), for a fair, apples-
    to-apples comparison under the identical risk measure."""
    hedge_pnl, total_cost, deltas = bs_delta_hedge_pnl(
        S, cfg.K, cfg.sigma, cfg.r, cfg.tau_grid, cost_rate=cost_rate)
    X = -payoff + hedge_pnl - total_cost

    L = -X
    lam = 1.0 / (1.0 - alpha)
    w_star = np.quantile(L, alpha)
    price = w_star + lam * np.maximum(L - w_star, 0).mean()

    return dict(X=X, deltas=deltas, cost=total_cost, price=price,
                mean_pnl=float(X.mean()), std_pnl=float(X.std()))


# =============================================================================
# SECTION 8: PLOTTING & REPORTING
# =============================================================================

def make_plots(cfg: MarketConfig, bs_price: float, results: dict, S_test: np.ndarray):
    """Produce all comparison figures and save them as PNG files."""
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10})

    # ---- 1) training curves, one subplot per scenario ----------------------
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (tag, r) in zip(axes, results.items()):
        h = r["history"]
        te, tl = zip(*h["train_loss"])
        ve, vl = zip(*h["val_loss"])
        ax.plot(te, tl, label="train loss")
        ax.plot(ve, vl, label="val loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("CVaR loss (= price estimate)")
        ax.set_title(r["title"])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Deep Hedging training convergence")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/training_curves.png")
    plt.close(fig)

    # ---- 2) P&L histograms, no transaction costs ---------------------------
    r = results["no_cost"]
    X_dh = r["deep_hedge"]["X"] + bs_price       # add back BS price for a like-for-like view
    X_bs = r["bs_delta"]["X"] + bs_price
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(-3, 3, 60)
    ax.hist(X_dh, bins=bins, alpha=0.55, density=True, label="Deep Hedge (PyTorch)", color="tab:blue")
    ax.hist(X_bs, bins=bins, alpha=0.55, density=True, label="Black-Scholes Delta", color="tab:orange")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Hedged P&L at maturity (Black-Scholes price charged)")
    ax.set_ylabel("density")
    ax.set_title("No transaction costs: Deep Hedge vs. Black-Scholes Delta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/pnl_hist_no_cost.png")
    plt.close(fig)

    # ---- 3) P&L histograms, with transaction costs --------------------------
    r = results["with_cost"]
    X_dh_c = r["deep_hedge"]["X"] + bs_price
    X_bs_c = r["bs_delta"]["X"] + bs_price
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo = min(X_dh_c.min(), X_bs_c.min()) - 0.5
    hi = max(X_dh_c.max(), X_bs_c.max()) + 0.5
    bins = np.linspace(lo, hi, 70)
    ax.hist(X_dh_c, bins=bins, alpha=0.55, density=True, label="Deep Hedge (cost-aware)", color="tab:blue")
    ax.hist(X_bs_c, bins=bins, alpha=0.55, density=True, label="Naive Black-Scholes Delta", color="tab:orange")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Hedged P&L at maturity (Black-Scholes price charged)")
    ax.set_ylabel("density")
    ax.set_title("With transaction costs: Deep Hedge vs. naive Black-Scholes Delta")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/pnl_hist_with_cost.png")
    plt.close(fig)

    # ---- 4) learned hedge ratio vs. Black-Scholes Delta, one mid-life day --
    day_idx = DAY_IDX_FOR_DELTA_PLOT
    tau_t = cfg.tau_grid[day_idx]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, tag, title in [(axes[0], "no_cost", "No transaction costs"),
                            (axes[1], "with_cost", "With transaction costs")]:
        r = results[tag]
        S_t = S_test[:, day_idx]
        net_delta = r["deep_hedge"]["deltas"][:, day_idx]
        # subsample for a readable scatter plot when the test set is huge
        n_plot = min(5000, len(S_t))
        plot_idx = np.random.default_rng(0).choice(len(S_t), n_plot, replace=False)
        S_grid = np.linspace(S_t.min(), S_t.max(), 200)
        bs_curve = bs_call_delta(S_grid, cfg.K, tau_t, cfg.sigma, cfg.r)
        ax.scatter(S_t[plot_idx], net_delta[plot_idx], s=6, alpha=0.2,
                   color="tab:blue", label="Deep Hedge output")
        ax.plot(S_grid, bs_curve, color="tab:orange", lw=2.2, label="Black-Scholes Delta")
        ax.set_xlabel(f"Spot $S_t$ at day {day_idx} (of {cfg.n_steps})")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Hedge ratio")
    axes[0].legend(fontsize=8)
    fig.suptitle("Learned hedge ratio vs. Black-Scholes Delta")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/delta_comparison.png")
    plt.close(fig)

    # ---- 5) mean transaction cost comparison --------------------------------
    r = results["with_cost"]
    mean_cost_dh = r["deep_hedge"]["cost"].mean()
    mean_cost_bs = r["bs_delta"]["cost"].mean()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["Deep Hedge", "Naive BS-Delta"], [mean_cost_dh, mean_cost_bs],
           color=["tab:blue", "tab:orange"])
    for i, v in enumerate([mean_cost_dh, mean_cost_bs]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.set_ylabel("Mean total transaction cost\nover the option's life")
    ax.set_title("Deep Hedge trades less when costly")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/cost_comparison.png")
    plt.close(fig)


def make_table(bs_price: float, results: dict):
    """Build and save the final summary results table."""
    rows = []
    for tag, r in results.items():
        for method_key, method_name in [("deep_hedge", "Deep Hedge (PyTorch)"),
                                         ("bs_delta", "Black-Scholes Delta")]:
            m = r[method_key]
            rows.append(dict(
                scenario=r["title"], method=method_name,
                cost_rate=r["cost_rate"], alpha=r["alpha"],
                price=round(float(m["price"]), 4),
                mean_pnl=round(float(m["mean_pnl"]), 4),
                std_pnl=round(float(m["std_pnl"]), 4),
            ))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/results_table.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nBlack-Scholes analytic (continuous-time) price: {bs_price:.4f}")
    return df


# =============================================================================
# SECTION 9: MAIN -- ties every section above together into one experiment
# =============================================================================

def main():
    print(f"Using device: {DEVICE}")
    if QUICK_TEST:
        print("QUICK_TEST = True -> running a tiny, fast sanity-check version.\n"
              "  Set QUICK_TEST = False at the top of this file for the full "
              "1,000,000 / 100,000 / 100,000 run.\n")

    # ---- 1) Market configuration & data -------------------------------------
    cfg = MarketConfig(S0=S0, K=K, sigma=SIGMA, r=R, n_days=N_DAYS)
    print(f"Market: S0={cfg.S0}  K={cfg.K}  sigma={cfg.sigma}  n_days={cfg.n_days}  "
          f"n_steps(=#networks)={cfg.n_steps}  T={cfg.T:.5f}y")
    print(f"Dataset sizes: train={N_TRAIN:,}  val={N_VAL:,}  test={N_TEST:,}")

    print("Simulating price paths ...")
    t_sim0 = time.time()
    S_train, S_val, S_test = make_datasets(cfg, N_TRAIN, N_VAL, N_TEST, seed=RANDOM_SEED)
    print(f"  done in {time.time() - t_sim0:.1f}s")

    payoff_train = np.maximum(S_train[:, -1] - cfg.K, 0.0)
    payoff_val = np.maximum(S_val[:, -1] - cfg.K, 0.0)
    payoff_test = np.maximum(S_test[:, -1] - cfg.K, 0.0)

    bs_price = bs_call_price(cfg.S0, cfg.K, cfg.T, cfg.sigma, cfg.r)
    print(f"Black-Scholes (continuous-time, analytic) price: {bs_price:.4f}")

    # ---- 2) Train + evaluate each scenario -----------------------------------
    scenario_titles = {
        "no_cost": "No transaction costs, CVaR(50%)",
        "with_cost": "1% transaction costs, CVaR(50%)",
    }

    results = {}
    for tag, cost_rate in COST_RATE_SCENARIOS.items():
        print(f"\n=== Training scenario '{tag}'  (cost_rate={cost_rate}, alpha={ALPHA}) ===")
        model, loss_fn, history = train_deep_hedge(
            S_train, payoff_train, S_val, payoff_val,
            n_steps=cfg.n_steps, cost_rate=cost_rate, alpha=ALPHA,
            n_hidden=HIDDEN_WIDTH, lr=LEARNING_RATE, batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS, device=DEVICE, seed=7,
        )

        dh_eval = evaluate(model, loss_fn, S_test, payoff_test, cost_rate, DEVICE)
        print(f"  [TEST] Deep Hedge   price={dh_eval['price']:.4f}  "
              f"mean={dh_eval['mean_pnl']:.4f}  std={dh_eval['std_pnl']:.4f}")

        bs_eval = evaluate_bs_delta_hedge(S_test, payoff_test, cfg, cost_rate, ALPHA)
        print(f"  [TEST] BS-Delta     price={bs_eval['price']:.4f}  "
              f"mean={bs_eval['mean_pnl']:.4f}  std={bs_eval['std_pnl']:.4f}")

        results[tag] = dict(title=scenario_titles[tag], cost_rate=cost_rate, alpha=ALPHA,
                             history=history, deep_hedge=dh_eval, bs_delta=bs_eval)

        torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_{tag}.pt")

    # ---- 3) Plots & results table ---------------------------------------------
    make_plots(cfg, bs_price, results, S_test)
    make_table(bs_price, results)
    print(f"\nAll outputs saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
