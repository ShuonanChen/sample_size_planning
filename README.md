# Sample Size Planning

How many samples (e.g., neurons) do I need to collect to be confident I've seen
all the categories (e.g., projection patterns) that exist — and to see each one
enough times?

This repo provides a simple Bayesian tool to answer two practical questions
from a small pilot dataset:

1. **How many distinct patterns are really out there?** Some patterns may exist
   but just haven't shown up yet in your sample.
2. **How many more samples should I collect** so that every pattern appears at
   least `T` times, with high confidence?

The full math is in [docs/num_pattern_estimate.pdf](docs/num_pattern_estimate.pdf).
This README gives the intuition.

---

## The setup (in plain words)

You ran a pilot experiment and observed `N` total samples falling into `K`
distinct categories, with counts `n = {n_1, n_2, ..., n_K}`.

Example (neurons and their projection targets):
- 100 neurons traced (`N = 100`)
- 20 project to spinal cord
- 30 project to cerebellum
- 25 project to thalamus
- 25 project to frontal cortex
- So you've seen `K = 4` patterns.

But maybe other projection patterns exist that you simply haven't sampled yet.
Call the true total number of patterns `S`. We have `S ≥ K`, and we want to
estimate `S`.

---

## The idea

We model the counts with a **Dirichlet–Multinomial** distribution:

- `S` is the (unknown) total number of patterns.
- `α` controls how evenly distributed the patterns are:
  - **Large α** → patterns occur at roughly equal rates (uniform).
  - **Small α** → a few patterns dominate, many are rare (sparse / skewed).
- The `S − K` patterns you didn't see contribute "zero count" terms in the
  likelihood, properly accounting for categories you might have missed.

We then jointly infer `S` and `α` from your observed counts using Bayes' rule:

```
P(S, α | observed counts)  ∝  P(observed counts | S, α) · P(S, α)
```

From this posterior we get:
- A **point estimate** (most likely value) of `S`.
- A **full posterior distribution** over `S`, so you can report uncertainty.

---

## Estimating how many more samples you need

Once we have a posterior over `S` and `α`, we ask: if we collect `N'` more
samples, will every pattern show up enough times? We pick the smallest `N'`
such that:

> With probability ≥ 1 − δ, every one of the `S` patterns is observed at
> least `T` times.

You choose:
- `T` — minimum observations per pattern you want (e.g., 5).
- `δ` — acceptable failure probability (e.g., 0.05 → 95% confidence).

The notebook returns the recommended total sample size.

---

## Why the prior matters: two regimes

The recommendation depends a lot on whether patterns are evenly or unevenly
distributed.

**Case 1 — evenly distributed (large α).** Observation: `n = {20, 30, 25, 25}`.
Roughly balanced → likely few unseen patterns → modest extra sampling needed.

**Case 2 — skewed (small α).** Observation: `n = {5, 3, 7, 85}`.
One pattern dominates → many rare patterns probably exist but were missed →
many more samples needed to be confident you've covered them all.

---

## Repo layout

```
src/sample_size_planning/
    inference.py   # joint_log_posterior, posterior_S_alpha
    summaries.py   # MAP / E[S] / credible interval
    plotting.py    # plot_results
    planning.py    # recommend_sample_size  (placeholder, coming soon)
estimate_patterns_from partial_observations.ipynb   # worked example
docs/num_pattern_estimate.pdf                       # full derivation
```

## Install

From the repo root:

```bash
pip install -e .
```

## Quick start (Python)

```python
import numpy as np
from sample_size_planning import posterior_S_alpha, plot_results

counts = np.array([20, 30, 25, 25])     # your pilot counts
res = posterior_S_alpha(counts)         # joint + marginal posteriors
plot_results(counts, res)               # plots + summary printout
```

Or open [estimate_patterns_from partial_observations.ipynb](estimate_patterns_from%20partial_observations.ipynb)
for the worked example.

## Roadmap

- `recommend_sample_size(res, T, delta)` — smallest additional `N'` such that
  every pattern is observed `≥ T` times with probability `≥ 1 − δ` (currently
  a stub in `planning.py`).
