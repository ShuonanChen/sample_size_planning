"""Bayesian estimation of the number of latent patterns and sample-size planning.

See the project README for the modeling details.
"""

from .inference import joint_log_posterior, posterior_S_alpha
from .summaries import summarize_posterior, PosteriorSummary
from .plotting import plot_results
from .planning import recommend_sample_size  # placeholder; see module

__all__ = [
    "joint_log_posterior",
    "posterior_S_alpha",
    "summarize_posterior",
    "PosteriorSummary",
    "plot_results",
    "recommend_sample_size",
]

__version__ = "0.1.0"
