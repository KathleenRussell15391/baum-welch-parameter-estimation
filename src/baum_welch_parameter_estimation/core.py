"""Core implementation of the Baum-Welch parameter estimation algorithm.

All math uses natural logarithms internally to avoid floating-point underflow
on long sequences (a real problem: a length-200 chain of probabilities \~0.1
underflows to 0.0 in plain float arithmetic). We convert back to linear space
only at the end of each forward/backward pass via a numerically-stable
log-sum-exp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


_LOG_TOL = -745.0  # log(0) equivalent; values below this are treated as -inf


def _log_sum_exp(values: Sequence[float]) -> float:
    """Numerically-stable log-sum-exp.

    Why: naive sum(exp(v)) overflows/underflows badly for chains longer than
    ~100 steps. The max-subtraction trick keeps every exponent <= 0.
    """
    if not values:
        return float("-inf")
    m = max(values)
    if m == float("-inf"):
        return float("-inf")
    total = 0.0
    for v in values:
        # math.exp can raise OverflowError if v - m is large positive, but
        # since m is the max, v - m <= 0, so exp is always <= 1. Safe.
        total += math.exp(v - m)
    return m + math.log(total)


def normalize(weights: List[float]) -> List[float]:
    """Normalize a list of non-negative weights into a probability distribution.

    Raises ValueError if the weights are all zero or any is negative.

    Why expose this: callers need to produce valid initial distributions and
    the estimator itself needs it internally; sharing one implementation avoids
    subtle inconsistencies (e.g., sum-to-1 vs max-to-1).
    """
    if not weights:
        raise ValueError("weights must be non-empty")
    for w in weights:
        if w < 0:
            raise ValueError(f"weights must be non-negative, got {w}")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return [w / total for w in weights]


def _log_normalize(log_weights: List[float]) -> List[float]:
    """Normalize log-space weights, returning linear probabilities."""
    z = _log_sum_exp(log_weights)
    if z == float("-inf"):
        raise ValueError("cannot normalize all-log-zero weights")
    return [math.exp(lw - z) for lw in log_weights]


def _mat_log_init(matrix: List[List[float]]) -> List[List[float]]:
    """Take a linear probability matrix and return its log, with -inf for zeros."""
    out: List[List[float]] = []
    for row in matrix:
        out_row: List[float] = []
        for p in row:
            if p <= 0.0:
                out_row.append(float("-inf"))
            else:
                out_row.append(math.log(p))
        out.append(out_row)
    return out


def _vec_log_init(vec: List[float]) -> List[float]:
    out: List[float] = []
    for p in vec:
        if p <= 0.0:
            out.append(float("-inf"))
        else:
            out.append(math.log(p))
    return out


def forward(
    observations: Sequence[int],
    log_pi: Sequence[float],
    log_A: Sequence[Sequence[float]],
    log_B: Sequence[Sequence[float]],
) -> Tuple[List[List[float]], float]:
    """Forward algorithm in log space.

    Args:
        observations: sequence of observed symbol indices (ints in [0, M)).
        log_pi: log initial-state distribution, length N.
        log_A: log transition matrix, N x N. log_A[i][j] = log P(j|i).
        log_B: log emission matrix, N x M. log_B[i][k] = log P(o=k|i).

    Returns:
        (alpha_log, log_likelihood) where alpha_log[t] is the list of
        log forward probabilities for each state at time t.

    Raises ValueError on empty sequence or invalid indices.
    """
    T = len(observations)
    if T == 0:
        raise ValueError("observations must be non-empty")
    N = len(log_pi)
    if N == 0:
        raise ValueError("log_pi must be non-empty")
    if len(log_A) != N or any(len(r) != N for r in log_A):
        raise ValueError("log_A must be N x N")
    if len(log_B) != N:
        raise ValueError("log_B must have N rows")
    M = len(log_B[0]) if N > 0 else 0
    if any(len(r) != M for r in log_B):
        raise ValueError("log_B rows must all have the same length")
    for o in observations:
        if not (0 <= o < M):
            raise ValueError(f"observation {o} out of range [0, {M})")

    alpha_log: List[List[float]] = []
    # t = 0
    first_obs = observations[0]
    alpha0 = [log_pi[i] + log_B[i][first_obs] for i in range(N)]
    alpha_log.append(alpha0)

    # t = 1 .. T-1
    for t in range(1, T):
        obs = observations[t]
        prev = alpha_log[t - 1]
        cur: List[float] = []
        for j in range(N):
            # log sum_i alpha[i] * A[i][j]
            terms = [prev[i] + log_A[i][j] for i in range(N)]
            s = _log_sum_exp(terms)
            cur.append(s + log_B[j][obs])
        alpha_log.append(cur)

    log_likelihood = _log_sum_exp(alpha_log[-1])
    return alpha_log, log_likelihood


def backward(
    observations: Sequence[int],
    log_A: Sequence[Sequence[float]],
    log_B: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Backward algorithm in log space.

    Args same as forward (without log_pi). Returns beta_log, the log backward
    probabilities. beta_log[t] has length N. By convention beta_log[T-1] is all 0.0
    (log 1).
    """
    T = len(observations)
    if T == 0:
        raise ValueError("observations must be non-empty")
    N = len(log_A)
    M = len(log_B[0]) if N > 0 else 0
    if N == 0:
        raise ValueError("log_A must be non-empty")
    for o in observations:
        if not (0 <= o < M):
            raise ValueError(f"observation {o} out of range [0, {M})")

    beta_log: List[List[float]] = [[0.0] * N for _ in range(T)]
    for t in range(T - 2, -1, -1):
        obs_next = observations[t + 1]
        nxt = beta_log[t + 1]
        cur: List[float] = []
        for i in range(N):
            terms = [log_A[i][j] + log_B[j][obs_next] + nxt[j] for j in range(N)]
            cur.append(_log_sum_exp(terms))
        beta_log[t] = cur
    return beta_log


def gamma(
    alpha_log: Sequence[Sequence[float]],
    beta_log: Sequence[Sequence[float]],
    log_likelihood: float,
) -> List[List[float]]:
    """Posterior state probabilities gamma[t][i] = P(state=i at time t | obs).

    Computed as alpha[t][i]*beta[t][i] / P(obs). In log space we subtract the
    log-likelihood. Returns linear probabilities.
    """
    T = len(alpha_log)
    N = len(alpha_log[0]) if T > 0 else 0
    out: List[List[float]] = []
    for t in range(T):
        row: List[float] = []
        for i in range(N):
            val = alpha_log[t][i] + beta_log[t][i] - log_likelihood
            if val < _LOG_TOL:
                row.append(0.0)
            else:
                row.append(math.exp(val))
        # Re-normalize to guard against tiny numerical drift; gamma rows must
        # sum to 1. Without this, accumulation over T steps can drift enough to
        # break the estimator's convergence check.
        s = sum(row)
        if s > 0:
            row = [r / s for r in row]
        out.append(row)
    return out


def xi(
    observations: Sequence[int],
    alpha_log: Sequence[Sequence[float]],
    beta_log: Sequence[Sequence[float]],
    log_A: Sequence[Sequence[float]],
    log_B: Sequence[Sequence[float]],
    log_likelihood: float,
) -> List[List[List[float]]]:
    """Joint posterior xi[t][i][j] = P(state=i at t, state=j at t+1 | obs).

    Returns a list of length T-1. Each entry is N x N. Linear probabilities.
    """
    T = len(observations)
    N = len(log_A)
    out: List[List[List[float]]] = []
    for t in range(T - 1):
        obs_next = observations[t + 1]
        mat: List[List[float]] = []
        denom_terms: List[float] = []
        for i in range(N):
            for j in range(N):
                term = (
                    alpha_log[t][i]
                    + log_A[i][j]
                    + log_B[j][obs_next]
                    + beta_log[t + 1][j]
                )
                denom_terms.append(term)
        denom = _log_sum_exp(denom_terms)
        # denom should equal log_likelihood; we use it directly to ensure the
        # matrix sums to exactly 1 regardless of small alpha/beta drift.
        for i in range(N):
            mat_row: List[float] = []
            for j in range(N):
                val = (
                    alpha_log[t][i]
                    + log_A[i][j]
                    + log_B[j][obs_next]
                    + beta_log[t + 1][j]
                    - denom
                )
                if val < _LOG_TOL:
                    mat_row.append(0.0)
                else:
                    mat_row.append(math.exp(val))
            mat.append(mat_row)
        out.append(mat)
    return out


@dataclass
class BaumWelchResult:
    """Result of a Baum-Welch estimation run.

    Attributes:
        transition: re-estimated transition matrix, N x N, rows sum to 1.
        emission: re-estimated emission matrix, N x M, rows sum to 1.
        log_likelihood: final sequence log-likelihood.
        iterations: number of EM iterations actually performed.
        converged: True if the log-likelihood improvement fell below tol.
    """

    transition: List[List[float]]
    emission: List[List[float]]
    log_likelihood: float
    iterations: int
    converged: bool


@dataclass
class BaumWelchEstimator:
    """Estimate HMM transition and emission probabilities via Baum-Welch.

    The initial-state distribution (pi) is held fixed. Estimating pi from a
    single sequence is statistically weak (only one sample of the initial
    state), so we treat it as known input rather than a free parameter. This is
    the stated design decision; see README.

    Attributes:
        n_states: number of hidden states N.
        n_symbols: number of emission symbols M.
        max_iter: maximum EM iterations.
        tol: convergence threshold on log-likelihood improvement.
        smoothing: small value added to expected counts before normalization.
            This prevents zero-probability entries from becoming permanent
            -inf log-traps (a real failure mode: once a transition is exactly
            zero, no future iteration can ever make it non-zero again, because
            xi for that pair is always zero).
    """

    n_states: int
    n_symbols: int
    max_iter: int = 100
    tol: float = 1e-5
    smoothing: float = 1e-8
    _log_A: List[List[float]] = field(default_factory=list, init=False, repr=False)
    _log_B: List[List[float]] = field(default_factory=list, init=False, repr=False)
    _log_pi: List[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_states < 1:
            raise ValueError("n_states must be >= 1")
        if self.n_symbols < 1:
            raise ValueError("n_symbols must be >= 1")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")
        if self.smoothing < 0:
            raise ValueError("smoothing must be non-negative")

    def _init_params(
        self,
        transition: Sequence[Sequence[float]],
        emission: Sequence[Sequence[float]],
        initial: Sequence[float],
    ) -> None:
        if len(transition) != self.n_states:
            raise ValueError("transition must have n_states rows")
        for row in transition:
            if len(row) != self.n_states:
                raise ValueError("transition rows must have length n_states")
        if len(emission) != self.n_states:
            raise ValueError("emission must have n_states rows")
        for row in emission:
            if len(row) != self.n_symbols:
                raise ValueError("emission rows must have length n_symbols")
        if len(initial) != self.n_states:
            raise ValueError("initial must have length n_states")
        # Validate that each row is a proper distribution; we allow zeros, which
        # become -inf in log space.
        for i, row in enumerate(transition):
            s = sum(row)
            if not (0.999 <= s <= 1.001):
                raise ValueError(f"transition row {i} sums to {s}, not 1")
            for p in row:
                if p < 0:
                    raise ValueError(f"transition row {i} has negative entry")
        for i, row in enumerate(emission):
            s = sum(row)
            if not (0.999 <= s <= 1.001):
                raise ValueError(f"emission row {i} sums to {s}, not 1")
            for p in row:
                if p < 0:
                    raise ValueError(f"emission row {i} has negative entry")
        s = sum(initial)
        if not (0.999 <= s <= 1.001):
            raise ValueError(f"initial sums to {s}, not 1")
        for p in initial:
            if p < 0:
                raise ValueError("initial has negative entry")
        self._log_A = _mat_log_init([list(r) for r in transition])
        self._log_B = _mat_log_init([list(r) for r in emission])
        self._log_pi = _vec_log_init(list(initial))

    def _reestimate(
        self, observations: Sequence[int]
    ) -> Tuple[float, List[List[float]], List[List[float]]]:
        alpha_log, ll = forward(observations, self._log_pi, self._log_A, self._log_B)
        beta = backward(observations, self._log_A, self._log_B)
        gam = gamma(alpha_log, beta, ll)
        xis = xi(observations, alpha_log, beta, self._log_A, self._log_B, ll)

        N = self.n_states
        M = self.n_symbols
        eps = self.smoothing

        # Transition: sum_t xi[t][i][j], normalized over j.
        new_A_num = [[eps] * N for _ in range(N)]
        for t_mat in xis:
            for i in range(N):
                for j in range(N):
                    new_A_num[i][j] += t_mat[i][j]
        new_log_A: List[List[float]] = []
        for i in range(N):
            denom = sum(new_A_num[i])
            if denom <= 0:
                # All transitions out of i had zero expected count across the
                # entire sequence. Keep the row as it was to avoid inventing a
                # new distribution from nothing.
                new_log_A.append(list(self._log_A[i]))
            else:
                new_log_A.append([math.log(v / denom) for v in new_A_num[i]])

        # Emission: sum over t where obs[t]==k of gamma[t][i].
        new_B_num = [[eps] * M for _ in range(N)]
        gam_sums = [eps] * N
        for t, o in enumerate(observations):
            for i in range(N):
                new_B_num[i][o] += gam[t][i]
                gam_sums[i] += gam[t][i]
        new_log_B: List[List[float]] = []
        for i in range(N):
            if gam_sums[i] <= 0:
                new_log_B.append(list(self._log_B[i]))
            else:
                new_log_B.append([math.log(v / gam_sums[i]) for v in new_B_num[i]])

        return ll, new_log_A, new_log_B

    def fit(
        self,
        observations: Sequence[int],
        transition: Sequence[Sequence[float]],
        emission: Sequence[Sequence[float]],
        initial: Sequence[float],
    ) -> BaumWelchResult:
        """Run Baum-Welch on a single observation sequence.

        Args:
            observations: list of observed symbol indices, each in [0, M).
            transition: initial guess for the N x N transition matrix.
            emission: initial guess for the N x M emission matrix.
            initial: fixed initial-state distribution (length N). Not updated.

        Returns:
            BaumWelchResult with re-estimated transition and emission.

        Notes:
            - The initial distribution is not re-estimated. If you need to
              estimate pi, use multiple sequences or supply a strong prior.
            - smoothing (default 1e-8) is added to every expected count. Without
              it, a single zero-probability transition locks that cell at zero
              forever (xi is zero for that pair at every step, so the numerator
              stays zero). The smoothing floor breaks this trap.
        """
        if not observations:
            raise ValueError("observations must be non-empty")
        self._init_params(transition, emission, initial)

        prev_ll = float("-inf")
        iterations = 0
        converged = False
        last_ll = float("-inf")

        for it in range(self.max_iter):
            iterations = it + 1
            ll, new_log_A, new_log_B = self._reestimate(observations)
            self._log_A = new_log_A
            self._log_B = new_log_B
            last_ll = ll
            if it > 0 and abs(ll - prev_ll) < self.tol:
                converged = True
                break
            prev_ll = ll

        # Convert final log params back to linear.
        final_A: List[List[float]] = []
        for i in range(self.n_states):
            row = [_log_normalize([self._log_A[i][j] for j in range(self.n_states)])]
            # _log_normalize on the full row guarantees row sums to 1 even after
            # small numerical drift across iterations.
            final_A.append(row[0])
        final_B: List[List[float]] = []
        for i in range(self.n_states):
            final_B.append(
                _log_normalize([self._log_B[i][k] for k in range(self.n_symbols)])
            )

        return BaumWelchResult(
            transition=final_A,
            emission=final_B,
            log_likelihood=last_ll,
            iterations=iterations,
            converged=converged,
        )
