"""Plotting utilities for the (S, alpha) posterior."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .summaries import summarize_posterior


def plot_results(counts, res, title: str = "", ci: float = 0.95, show: bool = True):
    """Plot marginal posteriors over S and alpha.

    Returns
    -------
    fig, axes, summary
    """
    summary = summarize_posterior(counts, res, ci=ci)

    S_values = res["S_values"]
    alpha_grid = res["alpha_grid"]
    P_S = res["P_S"]
    P_alpha = res["P_alpha"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(S_values, P_S, color="steelblue", lw=2)
    axes[0].axvspan(
        summary.CI_low, summary.CI_high, color="orange", alpha=0.15,
        label=f"{int(ci * 100)}% CI [{summary.CI_low}, {summary.CI_high}]",
    )
    axes[0].axvline(summary.MAP_S, color="tomato", ls="--",
                    label=f"MAP S = {summary.MAP_S}")
    axes[0].axvline(summary.E_S, color="darkorange", ls="-.",
                    label=f"E[S] = {summary.E_S:.2f}")
    axes[0].axvline(summary.K, color="gray", ls=":", label=f"K = {summary.K}")
    axes[0].set_xlabel("S")
    axes[0].set_ylabel("P(S | n, K)")
    axes[0].set_title(f"Marginal posterior over S  (N={summary.N}, K={summary.K})")
    axes[0].legend()

    axes[1].plot(alpha_grid, P_alpha, color="seagreen", lw=2)
    axes[1].axvline(summary.MAP_alpha, color="tomato", ls="--",
                    label=f"MAP alpha = {summary.MAP_alpha:.3g}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("alpha (log scale)")
    axes[1].set_ylabel("P(alpha | n, K)")
    axes[1].set_title("Marginal posterior over alpha")
    axes[1].legend()

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()

    print(f"counts = {list(np.asarray(counts))}")
    print(f"  MAP S     = {summary.MAP_S}")
    print(f"  E[S]      = {summary.E_S:.3f}")
    print(f"  {int(ci * 100)}% CI    = [{summary.CI_low}, {summary.CI_high}]")
    print(f"  MAP alpha = {summary.MAP_alpha:.4g}")

    return fig, axes, summary
