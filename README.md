# baum-welch-parameter-estimation

Estimates the emission and transition probabilities of a discrete hidden Markov model from an observed sequence using the Baum-Welch (expectation-maximization) algorithm. Standard library only, no dependencies.

## Usage

```python
from baum_welch_parameter_estimation import BaumWelchEstimator

estimator = BaumWelchEstimator(n_states=2, n_symbols=2, max_iter=100, tol=1e-5)
result = estimator.fit(
    observations=[0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
    transition=[[0.7, 0.3], [0.4, 0.6]],
    emission=[[0.9, 0.1], [0.2, 0.8]],
    initial=[0.5, 0.5],
)
print(result.transition)
print(result.emission)
print(result.log_likelihood, result.iterations, result.converged)
```

Exported names: `BaumWelchEstimator`, `BaumWelchResult` (returned by `fit`), `normalize`, `forward`, `backward`, `gamma`, `xi`. The lower-level functions operate in log space and are exposed for inspection and custom pipelines.

## Why this exists

The specific problem: you have a single observation sequence and a guess at the HMM structure, and you want to refine the transition and emission probabilities. The trade-off is that the initial-state distribution is **not** re-estimated. Estimating `pi` from one sequence uses exactly one sample of the initial state, which is statistically weak and tends to produce degenerate distributions. If you need to estimate `pi`, collect multiple sequences and fit them together (this library does not do multi-sequence training — that is a deliberate scope limit, not an oversight).

All probability arithmetic is done in log space with a log-sum-exp reduction. This avoids floating-point underflow on sequences longer than ~100 steps, which plain `alpha * A * B` arithmetic hits quickly.

## Edge cases you will hit

- **Zero probabilities lock.** If your initial guess contains an exact zero in the transition or emission matrix, that cell stays zero forever without intervention, because its expected count is always zero. The estimator adds a small `smoothing` value (default `1e-8`) to every expected count before normalization to break this trap. Set `smoothing=0` only if you know your initial guess has no zeros and you want exact zeros to persist.
- **Degenerate rows.** If a state is never visited (all its gammas are ~0 across the sequence), the estimator keeps that state's transition and emission rows unchanged rather than inventing a new distribution from nothing. This is the safest behavior for a single sequence.
- **Convergence.** `converged` is `True` when the log-likelihood improvement between iterations falls below `tol`. With `tol=0.0`, the estimator runs all `max_iter` iterations and reports `converged=False`.
