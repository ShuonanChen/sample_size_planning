"""Sample-size planning given a posterior over (S, alpha).

This module is a placeholder for the upcoming feature described in the README:
recommend the smallest additional sample size N' such that, with probability
>= 1 - delta, every one of the S patterns is observed at least T times.
"""

from __future__ import annotations


def recommend_sample_size(res, T: int = 5, delta: float = 0.05, **kwargs):
    """Recommend additional samples needed for coverage (NOT YET IMPLEMENTED).

    Parameters
    ----------
    res : dict
        Output of :func:`sample_size_planning.posterior_S_alpha`.
    T : int
        Minimum desired observations per pattern.
    delta : float
        Acceptable failure probability (1 - confidence).
    """
    raise NotImplementedError(
        "recommend_sample_size is planned but not yet implemented. "
        "See README ('Estimating how many more samples you need')."
    )
