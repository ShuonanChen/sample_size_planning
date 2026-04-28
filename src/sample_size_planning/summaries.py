"""Point and interval summaries derived from the posterior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PosteriorSummary:
    K: int
    N: int
    MAP_S: int
    E_S: float
    CI_low: int
    CI_high: int
    MAP_alpha: float


def summarize_posterior(counts, res, ci: float = 0.95) -> PosteriorSummary:
    """Compute MAP, posterior mean, and credible interval for S, plus MAP alpha.

    Parameters
    ----------
    counts : array-like
        Observed counts, used only for K and N.
    res : dict
        Output of :func:`sample_size_planning.posterior_S_alpha`.
    ci : float, default 0.95
        Central credible-interval mass for S.
    """
    counts = np.asarray(counts)
    S_values = res["S_values"]
    alpha_grid = res["alpha_grid"]
    P_S = res["P_S"]
    P_alpha = res["P_alpha"]

    pmf = P_S / P_S.sum()
    cdf = np.cumsum(pmf)
    tail = (1.0 - ci) / 2.0
    lo = int(S_values[np.searchsorted(cdf, tail)])
    hi = int(S_values[np.searchsorted(cdf, 1.0 - tail)])

    return PosteriorSummary(
        K=len(counts),
        N=int(np.sum(counts)),
        MAP_S=int(S_values[np.argmax(P_S)]),
        E_S=float((S_values * pmf).sum()),
        CI_low=lo,
        CI_high=hi,
        MAP_alpha=float(alpha_grid[np.argmax(P_alpha)]),
    )
