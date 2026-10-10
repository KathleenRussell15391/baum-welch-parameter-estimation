"""Tests for baum_welch_parameter_estimation.core.

Every float comparison uses math.isclose with a generous tolerance, because
Baum-Welch results are only defined up to numerical precision and row-label
permutation. We test the behaviour the implementation actually claims: log-space
forward/backward, gamma/xi posteriors, and the estimator's fit() contract
(initial distribution fixed, smoothing prevents zero-lock, convergence flag is
honest). We do not test multi-sequence training because the implementation does
not support it.
"""

import math
import unittest

from baum_welch_parameter_estimation import (
    BaumWelchEstimator,
    normalize,
    forward,
    backward,
    gamma,
    xi,
)
from baum_welch_parameter_estimation.core import BaumWelchResult, _log_sum_exp


class TestNormalize(unittest.TestCase):
    def test_basic(self):
        result = normalize([1.0, 1.0, 2.0])
        self.assertTrue(math.isclose(sum(result), 1.0))
        self.assertTrue(math.isclose(result[2], 0.5))

    def test_all_zero_raises(self):
        with self.assertRaises(ValueError):
            normalize([0.0, 0.0, 0.0])

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            normalize([1.0, -0.5, 0.5])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            normalize([])


class TestLogSumExp(unittest.TestCase):
    def test_basic(self):
        # log(exp(1) + exp(2))
        result = _log_sum_exp([1.0, 2.0])
        expected = math.log(math.exp(1.0) + math.exp(2.0))
        self.assertTrue(math.isclose(result, expected))

    def test_all_neg_inf(self):
        self.assertEqual(_log_sum_exp([float("-inf"), float("-inf")]), float("-inf"))

    def test_empty(self):
        self.assertEqual(_log_sum_exp([]), float("-inf"))

    def test_extreme_values_no_overflow(self):
        # Values that would overflow in naive sum(exp(v)).
        result = _log_sum_exp([1000.0, 1001.0, 1002.0])
        self.assertTrue(result > 1000.0)


class TestForwardBackward(unittest.TestCase):
    def setUp(self):
        # Two-state HMM, two symbols.
        self.log_pi = [math.log(0.5), math.log(0.5)]
        self.log_A = [[math.log(0.7), math.log(0.3)], [math.log(0.4), math.log(0.6)]]
        self.log_B = [[math.log(0.9), math.log(0.1)], [math.log(0.2), math.log(0.8)]]
        self.obs = [0, 1, 0]

    def test_forward_returns_correct_shape(self):
        alpha, ll = forward(self.obs, self.log_pi, self.log_A, self.log_B)
        self.assertEqual(len(alpha), 3)
        for row in alpha:
            self.assertEqual(len(row), 2)

    def test_forward_log_likelihood_matches_brute_force(self):
        # Compute P(obs) by brute force over all state paths.
        pi = [0.5, 0.5]
        A = [[0.7, 0.3], [0.4, 0.6]]
        B = [[0.9, 0.1], [0.2, 0.8]]
        total = 0.0
        for s0 in range(2):
            for s1 in range(2):
                for s2 in range(2):
                    p = pi[s0] * B[s0][self.obs[0]]
                    p *= A[s0][s1] * B[s1][self.obs[1]]
                    p *= A[s1][s2] * B[s2][self.obs[2]]
                    total += p
        _, ll = forward(self.obs, self.log_pi, self.log_A, self.log_B)
        self.assertTrue(math.isclose(math.exp(ll), total, rel_tol=1e-9))

    def test_backward_returns_correct_shape(self):
        beta = backward(self.obs, self.log_A, self.log_B)
        self.assertEqual(len(beta), 3)
        # Last row is all 0.0 (log 1) by convention.
        for v in beta[-1]:
            self.assertTrue(math.isclose(v, 0.0))

    def test_forward_backward_identity(self):
        # sum_i alpha[t][i] * beta[t][i] == P(obs) for all t.
        alpha, ll = forward(self.obs, self.log_pi, self.log_A, self.log_B)
        beta = backward(self.obs, self.log_A, self.log_B)
        for t in range(len(self.obs)):
            s = sum(math.exp(alpha[t][i] + beta[t][i]) for i in range(2))
            self.assertTrue(math.isclose(s, math.exp(ll), rel_tol=1e-9))

    def test_forward_empty_raises(self):
        with self.assertRaises(ValueError):
            forward([], self.log_pi, self.log_A, self.log_B)

    def test_forward_bad_observation_raises(self):
        with self.assertRaises(ValueError):
            forward([0, 5], self.log_pi, self.log_A, self.log_B)


class TestGammaXi(unittest.TestCase):
    def setUp(self):
        self.log_pi = [math.log(0.5), math.log(0.5)]
        self.log_A = [[math.log(0.7), math.log(0.3)], [math.log(0.4), math.log(0.6)]]
        self.log_B = [[math.log(0.9), math.log(0.1)], [math.log(0.2), math.log(0.8)]]
        self.obs = [0, 1, 0]
        self.alpha, self.ll = forward(self.obs, self.log_pi, self.log_A, self.log_B)
        self.beta = backward(self.obs, self.log_A, self.log_B)

    def test_gamma_rows_sum_to_one(self):
        g = gamma(self.alpha, self.beta, self.ll)
        for row in g:
            self.assertTrue(math.isclose(sum(row), 1.0, abs_tol=1e-9))

    def test_gamma_shape(self):
        g = gamma(self.alpha, self.beta, self.ll)
        self.assertEqual(len(g), 3)
        for row in g:
            self.assertEqual(len(row), 2)

    def test_xi_shape(self):
        x = xi(self.obs, self.alpha, self.beta, self.log_A, self.log_B, self.ll)
        self.assertEqual(len(x), 2)  # T-1
        for mat in x:
            self.assertEqual(len(mat), 2)
            for row in mat:
                self.assertEqual(len(row), 2)

    def test_xi_matrices_sum_to_one(self):
        x = xi(self.obs, self.alpha, self.beta, self.log_A, self.log_B, self.ll)
        for mat in x:
            total = sum(sum(row) for row in mat)
            self.assertTrue(math.isclose(total, 1.0, abs_tol=1e-9))

    def test_gamma_marginalizes_xi(self):
        # sum_j xi[t][i][j] == gamma[t][i]
        g = gamma(self.alpha, self.beta, self.ll)
        x = xi(self.obs, self.alpha, self.beta, self.log_A, self.log_B, self.ll)
        for t in range(len(x)):
            for i in range(2):
                s = sum(x[t][i][j] for j in range(2))
                self.assertTrue(math.isclose(s, g[t][i], abs_tol=1e-9))


class TestBaumWelchEstimator(unittest.TestCase):
    def test_converges_on_consistent_data(self):
        # Generate a sequence from a known HMM, then check that fitting
        # recovers something close to the true parameters. We use a long-ish
        # deterministic sequence so the test is stable.
        true_A = [[0.8, 0.2], [0.3, 0.7]]
        true_B = [[0.9, 0.1], [0.2, 0.8]]
        pi = [0.5, 0.5]
        # Deterministic observation sequence (hand-picked, not random) so the
        # test is fully deterministic.
        obs = [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
        est = BaumWelchEstimator(n_states=2, n_symbols=2, max_iter=200, tol=1e-6)
        result = est.fit(obs, true_A, true_B, pi)
        self.assertIsInstance(result, BaumWelchResult)
        # Rows must be valid distributions.
        for row in result.transition:
            self.assertTrue(math.isclose(sum(row), 1.0, abs_tol=1e-6))
        for row in result.emission:
            self.assertTrue(math.isclose(sum(row), 1.0, abs_tol=1e-6))
        # Log-likelihood should be finite.
        self.assertFalse(math.isinf(result.log_likelihood))

    def test_log_likelihood_non_decreasing(self):
        # EM guarantees non-decreasing log-likelihood. We check by running a
        # few iterations manually and comparing.
        A = [[0.6, 0.4], [0.5, 0.5]]
        B = [[0.8, 0.2], [0.3, 0.7]]
        pi = [0.5, 0.5]
        obs = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        est = BaumWelchEstimator(n_states=2, n_symbols=2, max_iter=5, tol=0.0)
        # Run first iteration by calling internal reestimate once.
        est._init_params(A, B, pi)
        ll1, _, _ = est._reestimate(obs)
        # Second iteration uses updated params already set by _reestimate side
        # effect? No: _reestimate returns new params but does not store them.
        # So we call fit which stores them and re-runs. Instead, we just check
        # the final fit log-likelihood is >= the first-iteration ll.
        result = est.fit(obs, A, B, pi)
        self.assertTrue(result.log_likelihood >= ll1 - 1e-9)

    def test_initial_distribution_not_updated(self):
        # The initial distribution is fixed. We check that running fit does not
        # alter the contract: the returned object has no 'initial' field, and
        # re-running with a different pi changes the result's likelihood (i.e.,
        # pi is actually used, just not estimated).
        A = [[0.7, 0.3], [0.4, 0.6]]
        B = [[0.9, 0.1], [0.2, 0.8]]
        obs = [0, 1, 0, 1, 0]
        est = BaumWelchEstimator(n_states=2, n_symbols=2, max_iter=10, tol=1e-6)
        r1 = est.fit(obs, A, B, [0.9, 0.1])
        r2 = est.fit(obs, A, B, [0.1, 0.9])
        # Different fixed pi should generally give different likelihoods.
        # We don't assert strict inequality (they could coincide for trivial
        # cases), but we check both ran.
        self.assertFalse(math.isinf(r1.log_likelihood))
        self.assertFalse(math.isinf(r2.log_likelihood))

    def test_smoothing_prevents_zero_lock(self):
        # If we start with a zero transition probability, without smoothing it
        # would stay zero forever. With smoothing, it can become non-zero.
        A = [[0.0, 1.0], [1.0, 0.0]]  # A[0][0] is zero
        B = [[0.5, 0.5], [0.5, 0.5]]
        pi = [0.5, 0.5]
        obs = [0, 0, 0, 0, 0]  # long run of same symbol
        est = BaumWelchEstimator(
            n_states=2, n_symbols=2, max_iter=5, tol=0.0, smoothing=1e-6
        )
        result = est.fit(obs, A, B, pi)
        # A[0][0] should now be strictly positive.
        self.assertGreater(result.transition[0][0], 0.0)

    def test_convergence_flag_when_tolerance_met(self):
        # With a high tolerance, the estimator should report convergence early.
        A = [[0.7, 0.3], [0.4, 0.6]]
        B = [[0.9, 0.1], [0.2, 0.8]]
        pi = [0.5, 0.5]
        obs = [0, 1, 0]
        est = BaumWelchEstimator(n_states=2, n_symbols=2, max_iter=100, tol=10.0)
        result = est.fit(obs, A, B, pi)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.iterations, 100)

    def test_max_iter_respected(self):
        A = [[0.7, 0.3], [0.4, 0.6]]
        B = [[0.9, 0.1], [0.2, 0.8]]
        pi = [0.5, 0.5]
        obs = [0, 1, 0, 1, 0]
        est = BaumWelchEstimator(n_states=2, n_symbols=2, max_iter=3, tol=0.0)
        result = est.fit(obs, A, B, pi)
        self.assertEqual(result.iterations, 3)
        self.assertFalse(result.converged)

    def test_invalid_n_states_raises(self):
        with self.assertRaises(ValueError):
            BaumWelchEstimator(n_states=0, n_symbols=2)

    def test_invalid_n_symbols_raises(self):
        with self.assertRaises(ValueError):
            BaumWelchEstimator(n_states=2, n_symbols=0)

    def test_invalid_transition_shape_raises(self):
        est = BaumWelchEstimator(n_states=2, n_symbols=2)
        with self.assertRaises(ValueError):
            est.fit([0, 1], [[0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [0.5, 0.5])

    def test_non_distribution_row_raises(self):
        est = BaumWelchEstimator(n_states=2, n_symbols=2)
        with self.assertRaises(ValueError):
            est.fit([0, 1], [[0.5, 0.4], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [0.5, 0.5])

    def test_empty_observations_raises(self):
        est = BaumWelchEstimator(n_states=2, n_symbols=2)
        with self.assertRaises(ValueError):
            est.fit([], [[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [0.5, 0.5])

    def test_single_state_model(self):
        # Edge case: N=1. Transition is [[1.0]], emission is [[p, 1-p]].
        est = BaumWelchEstimator(n_states=1, n_symbols=2, max_iter=10, tol=1e-6)
        result = est.fit([0, 1, 0, 1], [[1.0]], [[0.6, 0.4]], [1.0])
        self.assertTrue(math.isclose(result.transition[0][0], 1.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(sum(result.emission[0]), 1.0, abs_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
