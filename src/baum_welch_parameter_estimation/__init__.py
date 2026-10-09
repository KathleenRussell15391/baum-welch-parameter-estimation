"""Package exports for baum_welch_parameter_estimation.

This library estimates the emission and transition probabilities of a discrete
hidden Markov model (HMM) from an observed emission sequence, using the
expectation-maximization (Baum-Welch) algorithm.

Design choice: we adopt the convention that the model is homogeneous (one set
of parameters for all time steps) and that the initial-state distribution is
fixed and supplied by the caller, not re-estimated. Estimating the initial
distribution from a single sequence is statistically weak and numerically
unstable, so we keep it fixed. This is the single, clearly-stated interpretation
of the brief.
"""

from .core import BaumWelchEstimator, normalize, forward, backward, gamma, xi

__all__ = [
    "BaumWelchEstimator",
    "normalize",
    "forward",
    "backward",
    "gamma",
    "xi",
]
