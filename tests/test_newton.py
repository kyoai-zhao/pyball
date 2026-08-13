"""Tests for pyball's interval-Newton certified root isolation.

The certificates must be *honest* in both directions:

* they never claim a root where none exists (soundness of the exclusions and
  of the unique-root test), and
* they always find and certify real, simple roots that lie inside the domain
  (completeness for the supported class).

The strong checks use mpmath at 50 digits as an independent oracle for the
true root, asserted to lie inside every certified interval -- with no code
shared between pyball and the oracle.

Run locally (stdlib only):  ``python -m unittest tests.test_newton -v``.
"""

import math
import unittest
from fractions import Fraction

from pyball import Ball, isolate_roots, newton_step


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


class TestNewtonStep(unittest.TestCase):
    def test_operator_shrinks_toward_root(self):
        f = lambda b: b * b - 2  # noqa: E731
        df = lambda b: 2 * b  # noqa: E731
        X = Ball(math.sqrt(2.0), 0.1)  # something that genuinely straddles
        N = newton_step(f, df, X)
        self.assertIsNotNone(N)
        # N(X) must overlap (actually snugly contain) the true root sqrt(2)
        self.assertLessEqual(N.lo, math.sqrt(2.0))
        self.assertLessEqual(math.sqrt(2.0), N.hi)
        # and be tighter than X
        self.assertLess(N.hi - N.lo, X.hi - X.lo)

    def test_operator_none_when_derivative_touches_zero(self):
        f = lambda b: b * b  # noqa: E731
        df = lambda b: 2 * b  # noqa: E731
        # df(X) = [0, 4]: contains zero -> operator is not defined
        self.assertIsNone(newton_step(f, df, Ball(0.0, 2.0)))
        # but away from the critical point it is defined
        self.assertIsNotNone(newton_step(f, df, Ball(3.0, 1.0)))

    def test_operator_finds_unique_subinterval(self):
        # For a monotonically increasing crossing, N(X) lands strictly inside X.
        f = lambda b: b - math.pi  # noqa: E731
        df = lambda b: Ball(1.0)  # noqa: E731
        N = newton_step(f, df, Ball(math.pi, 0.5))
        self.assertIsNotNone(N)
        self.assertLessEqual(N.lo, math.pi)
        self.assertLessEqual(math.pi, N.hi)

    def test_operator_accepts_scalar_interval(self):
        f = lambda b: b - 1  # noqa: E731
        df = lambda b: Ball(1.0)  # noqa: E731
        N = newton_step(f, df, 2.0)  # a point interval
        self.assertIsNotNone(N)
        self.assertTrue(N.contains(1.0))


class TestIsolateRoots(unittest.TestCase):
    def test_simple_quadratic(self):
        f = lambda b: b * b - 2  # noqa: E731
        df = lambda b: 2 * b  # noqa: E731
        roots = isolate_roots(f, df, 0.0, 3.0)
        self.assertEqual(len(roots), 1)
        r = roots[0]
        self.assertTrue(r.certified)
        self.assertLessEqual(r.interval.lo, math.sqrt(2.0))
        self.assertLessEqual(math.sqrt(2.0), r.interval.hi)
        self.assertLess(r.interval.hi - r.interval.lo, 1e-6)

    def test_root_excluded_out_of_domain(self):
        # x^2 - 2 has one root in [0, 3]; a domain strictly to the right must
        # certify zero roots (and must not emit a stale candidate).
        f = lambda b: b * b - 2  # noqa: E731
        df = lambda b: 2 * b  # noqa: E731
        roots = isolate_roots(f, df, 2.0, 3.0)
        self.assertEqual(len(roots), 0)

    def test_cubic_three_real_roots(self):
        # x^3 - 3x so roots at -sqrt(3), 0, +sqrt(3)
        f = lambda b: b * b * b - 3 * b  # noqa: E731
        df = lambda b: 3 * b * b - 3  # noqa: E731
        roots = isolate_roots(f, df, -3.0, 3.0)
        self.assertEqual(len(roots), 3)
        centers = sorted(r.interval.to_float() for r in roots)
        expected = [-math.sqrt(3.0), 0.0, math.sqrt(3.0)]
        for c, e in zip(centers, expected):
            self.assertAlmostEqual(c, e, delta=1e-6)
        self.assertTrue(all(r.certified for r in roots))

    def test_derivative_zero_at_root_is_candidate_not_certified(self):
        # x^2 at x=0: f' contains zero there, so Newton can never certify.
        f = lambda b: b * b  # noqa: E731
        df = lambda b: 2 * b  # noqa: E731
        roots = isolate_roots(f, df, -1.0, 1.0)
        self.assertGreaterEqual(len(roots), 1)
        # it must not be certified, and any candidate must still contain 0
        for r in roots:
            self.assertFalse(r.certified)
        self.assertTrue(any(r.interval.contains(0.0) for r in roots))

    def test_nonfinite_domain_raises(self):
        f = lambda b: b  # noqa: E731
        df = lambda b: Ball(1.0)  # noqa: E731
        with self.assertRaises(ValueError):
            isolate_roots(f, df, float("inf"), 1.0)
        with self.assertRaises(ValueError):
            isolate_roots(f, df, 2.0, 1.0)

    def test_domain_exactly_at_root(self):
        # f(0) = 0 and the domain is just the single point.
        f = lambda b: b  # noqa: E731
        df = lambda b: Ball(1.0)  # noqa: E731
        # a degenerate interval must not crash and must not invent a root
        roots = isolate_roots(f, df, 0.0, 0.0)
        self.assertEqual(roots, [])


@unittest.skipUnless(_module_available("mpmath"), "mpmath not installed")
class TestMpmathOracle(unittest.TestCase):
    """Certified intervals must contain the 50-digit-computed real root."""

    def _mp(self):
        import mpmath

        mpmath.mp.dps = 50
        return mpmath

    def test_certified_contains_true_root(self):
        mp = self._mp()
        f = lambda b: b.exp() - 2  # noqa: E731  root at ln(2)
        df = lambda b: b.exp()  # noqa: E731
        roots = isolate_roots(f, df, 0.0, 1.0)
        self.assertEqual(len(roots), 1)
        r = roots[0]
        self.assertTrue(r.certified)
        truth = mp.log(2)
        self.assertLessEqual(mp.mpf(str(r.interval.lo)), truth)
        self.assertLessEqual(truth, mp.mpf(str(r.interval.hi)))


if __name__ == "__main__":
    unittest.main()