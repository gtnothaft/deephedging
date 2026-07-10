"""Semi-analytical Heston pricing and delta utilities.

The implementation uses the Heston (1993) characteristic-function formula and
Gauss-Laguerre quadrature. It is intended for project-scale benchmarking rather
than production trading analytics.

Reference: Steven L. Heston (1993), A Closed-Form Solution for Options with
Stochastic Volatility with Applications to Bond and Currency Options.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.laguerre import laggauss


@lru_cache(maxsize=16)
def _laguerre_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached Gauss-Laguerre nodes and weights."""
    x, w = laggauss(n)
    return x.astype(float), w.astype(float)


def _heston_cf(
    u: np.ndarray,
    S: np.ndarray,
    v: np.ndarray,
    tau: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> np.ndarray:
    """Characteristic function of log(S_T) under the risk-neutral Heston model.

    Dynamics used by this pricer:
        dS_t = r S_t dt + sqrt(v_t) S_t dW_t^S
        dv_t = kappa(theta - v_t)dt + xi sqrt(v_t)dW_t^v
        corr(dW^S, dW^v) = rho

    Parameters are broadcast so that ``u`` can have shape [n_quad, 1] and
    ``S``/``v`` can have shape [1, n_paths].
    """
    i = 1j
    S = np.maximum(np.asarray(S, dtype=float), 1e-12)
    v = np.maximum(np.asarray(v, dtype=float), 0.0)
    x = np.log(S)

    # Heston's "little trap"-compatible form.
    d = np.sqrt((rho * xi * i * u - kappa) ** 2 + xi**2 * (i * u + u**2))
    g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)
    exp_dt = np.exp(-d * tau)

    C = (
        i * u * (x + r * tau)
        + (kappa * theta / xi**2)
        * ((kappa - rho * xi * i * u - d) * tau - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
    )
    D = ((kappa - rho * xi * i * u - d) / xi**2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    return np.exp(C + D * v)


def heston_call_price_delta_numpy(
    S,
    v,
    K: float,
    tau: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    *,
    n_quad: int = 64,
    chunk_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Heston European call price and spot delta.

    Returns
    -------
    price, delta : np.ndarray
        Arrays broadcast to the shape of ``S`` and ``v``.

    Notes
    -----
    The call price is ``C = S P1 - K exp(-r tau) P2``.  For this formula the
    spot delta is ``P1`` when current variance is held fixed.  That is the
    model-consistent Heston stock delta used as the classical Heston hedge.
    """
    S_arr = np.asarray(S, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    out_shape = np.broadcast_shapes(S_arr.shape, v_arr.shape)
    S_flat = np.broadcast_to(S_arr, out_shape).reshape(-1)
    v_flat = np.broadcast_to(v_arr, out_shape).reshape(-1)

    price = np.empty_like(S_flat, dtype=float)
    delta = np.empty_like(S_flat, dtype=float)

    # At expiry, the price is the payoff and the delta is the payoff slope.
    if tau <= 1e-10:
        price[:] = np.maximum(S_flat - K, 0.0)
        delta[:] = (S_flat > K).astype(float)
        return price.reshape(out_shape), delta.reshape(out_shape)

    # If vol-of-vol is effectively zero, fall back to the constant-vol limit.
    # This avoids numerical division by xi^2 in nearly Black-Scholes cases.
    if abs(xi) < 1e-8:
        from scipy.stats import norm

        sigma = np.sqrt(np.maximum(v_flat, 1e-16))
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(np.maximum(S_flat, 1e-12) / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau
        price[:] = S_flat * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        delta[:] = norm.cdf(d1)
        return price.reshape(out_shape), delta.reshape(out_shape)

    nodes, weights = _laguerre_nodes_weights(n_quad)
    u = nodes[:, None]
    w = weights[:, None]
    i = 1j
    logK = np.log(K)

    for start in range(0, S_flat.size, chunk_size):
        stop = min(start + chunk_size, S_flat.size)
        S_chunk = S_flat[start:stop][None, :]
        v_chunk = np.maximum(v_flat[start:stop], 0.0)[None, :]

        phi_minus_i = _heston_cf(-1j, S_chunk, v_chunk, tau, r, kappa, theta, xi, rho)
        phi1 = _heston_cf(u - 1j, S_chunk, v_chunk, tau, r, kappa, theta, xi, rho)
        phi2 = _heston_cf(u, S_chunk, v_chunk, tau, r, kappa, theta, xi, rho)

        integrand1 = np.real(np.exp(-i * u * logK) * phi1 / (i * u * phi_minus_i))
        integrand2 = np.real(np.exp(-i * u * logK) * phi2 / (i * u))

        # Gauss-Laguerre integrates int_0^inf exp(-x) g(x) dx, so for
        # int f(x) dx we use g(x)=exp(x) f(x).
        P1 = 0.5 + (1.0 / np.pi) * np.sum(w * np.exp(u) * integrand1, axis=0)
        P2 = 0.5 + (1.0 / np.pi) * np.sum(w * np.exp(u) * integrand2, axis=0)

        delta[start:stop] = np.clip(P1, 0.0, 1.0)
        price[start:stop] = S_flat[start:stop] * P1 - K * np.exp(-r * tau) * P2

    return price.reshape(out_shape), delta.reshape(out_shape)


def initial_heston_price(cfg, *, n_quad: int = 96) -> float:
    """Initial Heston model price for the project ATM call."""
    v0 = cfg.sigma**2
    price, _ = heston_call_price_delta_numpy(
        cfg.S0,
        v0,
        cfg.K,
        cfg.T,
        cfg.r,
        cfg.heston_kappa,
        cfg.heston_theta,
        cfg.heston_xi,
        cfg.heston_rho,
        n_quad=n_quad,
    )
    return float(np.asarray(price).item())


def heston_delta_torch(
    S_t,
    v_t,
    tau: float,
    cfg,
    *,
    n_quad: int = 64,
    chunk_size: int = 20_000,
):
    """Compute model-consistent Heston delta and return a torch tensor.

    This is not differentiable with respect to PyTorch tensors; it is meant for
    evaluating the classical Heston delta benchmark, not for training the NN.
    """
    import torch

    device = S_t.device
    S_np = S_t.detach().cpu().numpy()
    v_np = v_t.detach().cpu().numpy()
    _, delta_np = heston_call_price_delta_numpy(
        S_np,
        v_np,
        cfg.K,
        tau,
        cfg.r,
        cfg.heston_kappa,
        cfg.heston_theta,
        cfg.heston_xi,
        cfg.heston_rho,
        n_quad=n_quad,
        chunk_size=chunk_size,
    )
    return torch.as_tensor(delta_np, dtype=S_t.dtype, device=device)
