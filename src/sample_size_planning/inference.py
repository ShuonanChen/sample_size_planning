"""Joint posterior over (S, alpha) for the Dirichlet-Multinomial model."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import gammaln

# np.trapz was renamed to np.trapezoid in NumPy 2.0
_trapz = getattr(np, "trapezoid", np.trapz)


def joint_log_posterior(
    counts: np.ndarray,
    S_values: np.ndarray,
    alpha_grid: np.ndarray,
) -> np.ndarray:
    """Log of P(S, alpha | n, K) up to a constant, evaluated on a grid.

    Parameters
    ----------
    counts : array-like of int, shape (K,)
        Observed per-category counts (only the K categories that were seen).
    S_values : array-like of int, shape (nS,)
        Candidate values of S (total number of categories), with S >= K.
    alpha_grid : array-like of float, shape (nA,)
        Candidate values of the symmetric Dirichlet concentration alpha.

    Returns
    -------
    log_joint : ndarray, shape (nS, nA)
        Unnormalized log joint posterior with a flat prior on (S, alpha).
    """
    counts = np.asarray(counts)
    K = len(counts)
    N = counts.sum()

    S = np.asarray(S_values, dtype=float)[:, None]   # (nS, 1)
    a = np.asarray(alpha_grid, dtype=float)[None, :]  # (1, nA)

    # log C(S, K)
    log_binom = gammaln(S + 1) - gammaln(K + 1) - gammaln(S - K + 1)

    # Dirichlet-Multinomial normalization terms
    log_norm = gammaln(S * a) - gammaln(N + S * a)

    # sum_k [lgamma(alpha + n_k) - lgamma(alpha)], shape (1, nA)
    sum_obs = (gammaln(a[..., None] + counts) - gammaln(a[..., None])).sum(axis=-1)

    return log_binom + log_norm + sum_obs


def posterior_S_alpha(
    counts: np.ndarray,
    S_max: Optional[int] = None,
    alpha_grid: Optional[np.ndarray] = None,
) -> dict:
    """Joint inference on (S, alpha) and marginals, with a flat prior.

    Parameters
    ----------
    counts : array-like of int, shape (K,)
        Observed per-category counts.
    S_max : int, optional
        Largest S to evaluate. Defaults to ``max(10*K, 50)``.
    alpha_grid : array-like, optional
        Grid for alpha. Defaults to ``np.logspace(-2, 2, 200)``.

    Returns
    -------
    dict with keys:
        ``S_values``, ``alpha_grid``, ``joint``, ``P_S``, ``P_alpha``.
        ``joint`` is normalized so that ``sum_S trapz(joint, alpha) = 1``.
        ``P_S`` is a discrete pmf over ``S_values``.
        ``P_alpha`` is a density on ``alpha_grid`` (trapezoid-normalized).
    """
    counts = np.asarray(counts)
    K = len(counts)
    if S_max is None:
        S_max = max(10 * K, 50)
    if alpha_grid is None:
        alpha_grid = np.logspace(-2, 2, 200)
    alpha_grid = np.asarray(alpha_grid, dtype=float)

    S_values = np.arange(K, S_max + 1)

    log_joint = joint_log_posterior(counts, S_values, alpha_grid)

    # numeric stabilization then normalize
    joint = np.exp(log_joint - log_joint.max())
    Z = _trapz(joint, alpha_grid, axis=1).sum()
    joint /= Z

    P_S = _trapz(joint, alpha_grid, axis=1)            # discrete pmf
    P_alpha = joint.sum(axis=0)                         # density
    P_alpha /= _trapz(P_alpha, alpha_grid)

    return {
        "S_values": S_values,
        "alpha_grid": alpha_grid,
        "joint": joint,
        "P_S": P_S,
        "P_alpha": P_alpha,
    }
