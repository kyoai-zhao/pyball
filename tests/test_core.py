"""Tests for pyball's scalar Ball core.

Strategy: every property is checked on both an *exact* (Fraction/zero-radius)
regime and a *floating* regime, because radical endpoint rounding only exists
in the latter.  The floating regime compares against an independent oracle:

* ``decimal.Decimal(float)`` is the *exact* value of the binary float, so
  ``+ - * / sqrt`` enclosures are checked against exact rational arithmetic
  done at 60 significant digits -- a genuine soundness test with no shared
  code with pyball.
* mpmath (when installed) serves as a 50-digit truth oracle for the
  transcendental enclosures.  Those tests are skipped locally when mpmath is
  missing but run in CI.

Run locally (stdlib only):  ``python -m unittest -v`` from the pyball dir.
Also runs under pytest unchanged.
"""

import math
import unittest
from decimal import Decimal, getcontext
from fractions import Fraction

from pyball import Ball, INF, NaN
from pyball import prove_positive_negative

getcontext().prec = 60

FLOAT_CASES = [
    (0.1, 10 ** (-20)),
    (1e3, 1e-9),
    (3.141592653589793, 1e-15),
    (2.0, 0.5),
    (-2.5, 0.5),
    (2 ** -100, 2 ** -120),
    (1e200, 1e190),
]


def _dec(x) -> Decimal:
    if isinstance(x, str):
        return Decimal(x)  # already a decimal digit string
    return Decimal(repr(x))


def _dec_ball(b: Ball):
    """Return (lo, hi) as Decimals for exact value containment checks."""
    return _dec(b.lo), _dec(b.hi)


class TestSoundness(unittest.TestCase):
    def test_lo_hi_enclose_mid(self):
        for mid, rad in FLOAT_CASES:
            b = Ball(mid, rad)
            with self.subTest(mid=mid, rad=rad):
                self.assertLessEqual(b.lo, b.to_float())
                self.assertLessEqual(b.to_float(), b.hi)
                self.assertLessEqual(b.lo, b.hi)

    def test_lo_hi_outward_rounded(self):
        for mid, rad in FLOAT_CASES:
            b = Ball(mid, rad)
            with self.subTest(mid=mid, rad=rad):
                self.assertEqual(b.lo, math.nextafter(mid - rad, -INF))
                self.assertEqual(b.hi, math.nextafter(mid + rad, INF))

    def test_exact_mid_keeps_fraction(self):
        b = Ball(Fraction(1, 3))
        self.assertEqual(b.mid, Fraction(1, 3))
        self.assertTrue(b.is_exact)
        self.assertEqual(b.rad, 0.0)

    def test_negative_rad_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            Ball(1.0, -0.1)
        with self.assertRaises(ValueError):
            Ball(0.0, math.nan)
        with self.assertRaises(ValueError):
            Ball(math.nan)

    def test_bool_rejected(self):
        with self.assertRaises(TypeError):
            Ball(True)


class TestExactArithmetic(unittest.TestCase):
    def test_exact_arithmetic_stays_exact(self):
        a = Ball(Fraction(1, 3))
        b = Ball(Fraction(1, 3))
        self.assertTrue((a + b).is_exact)
        self.assertEqual((a * b).mid, Fraction(1, 9))
        self.assertEqual((a - b).mid, 0)
        self.assertEqual((a / b).mid, 1)

    def test_float_arithmetic_not_exact(self):
        a = Ball(0.1)
        self.assertFalse(a.is_exact)
        self.assertFalse((a + 0.2).is_exact)


class TestBasicArithmeticDecimalOracle(unittest.TestCase):
    """Enclosure against exact Decimal arithmetic -- no code shared with Ball."""

    def _check(self, b: Ball, truth: Decimal):
        lo, hi = _dec_ball(b)
        self.assertLessEqual(lo, truth)
        self.assertLessEqual(truth, hi)

    def test_add(self):
        for mid, rad in FLOAT_CASES:
            with self.subTest(mid=mid, rad=rad):
                s = Ball(mid, rad) + Ball(0.2, 1e-20)
                self._check(s, _dec(mid) + _dec(0.2))

    def test_sub(self):
        for mid, rad in FLOAT_CASES:
            with self.subTest(mid=mid, rad=rad):
                s = Ball(mid, rad) - Ball(0.1, 1e-20)
                self._check(s, _dec(mid) - _dec(0.1))

    def test_mul(self):
        a = Ball(0.1, 1e-30)
        b = Ball(0.3, 1e-30)
        self._check(a * b, _dec(0.1) * _dec(0.3))
        for mid, rad in FLOAT_CASES:
            with self.subTest(mid=mid, rad=rad):
                self._check(Ball(mid, rad) * Ball(2.0, 0.0), _dec(mid) * Decimal(2))

    def test_div(self):
        a = Ball(1.0, 1e-30)
        b = Ball(3.0, 1e-30)
        self._check(a / b, _dec(1.0) / _dec(3.0))
        q = Ball(1.0, 0.0) / Ball(7.0, 0.0)
        self._check(q, Decimal(1) / Decimal(7))

    def test_sqrt(self):
        r = Ball(2.0, 1e-30).sqrt()
        self._check(r, _dec(2).sqrt())

    def test_div_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            Ball(1.0, 0.1) / 0.0
        with self.assertRaises(ZeroDivisionError):
            Ball(1.0) / Ball(0.0, 0.1)

    def test_sqrt_negative_raises(self):
        with self.assertRaises(ValueError):
            Ball(-1.0).sqrt()


class TestPow(unittest.TestCase):
    def test_pow_exact_small_int(self):
        b = Ball(Fraction(2))
        self.assertEqual((b ** 2).mid, 4)
        self.assertEqual((b ** -2).mid, Fraction(1, 4))

    def test_pow_zero_and_one(self):
        self.assertEqual(Ball(3.3) ** 0, Ball(1))
        self.assertEqual(Ball(3.3) ** 1, Ball(3.3))

    def test_pow_float_exponent_encloses_true(self):
        r = Ball(2.0, 1e-15) ** 0.5
        self.assertLessEqual(r.lo, math.sqrt(2.0))
        self.assertLessEqual(math.sqrt(2.0), r.hi)

    def test_pow_fractional_exact_base(self):
        r = Ball(Fraction(9)) ** Fraction(1, 2)
        self.assertTrue(r.contains(3.0))

    def test_pow_negative_base_fractional_raises(self):
        with self.assertRaises(ValueError):
            Ball(-1.0) ** 0.5


class TestTranscendentals(unittest.TestCase):
    def _check(self, b: Ball, truth_low: Decimal, truth_high: Decimal):
        self.assertLessEqual(_dec(b.lo), truth_low)
        self.assertLessEqual(truth_high, _dec(b.hi))

    def test_lipschitz_widening_monotone(self):
        """Widening the input must not shrink the output enclosure."""
        for fn in ("exp", "log", "sin", "cos", "atan", "sqrt"):
            with self.subTest(fn=fn):
                x0 = Ball(0.3, 1e-9)
                x1 = Ball(0.3, 1e-6)
                a, b = getattr(x0, fn)(), getattr(x1, fn)()
                self.assertGreaterEqual(abs(b.hi - b.lo), abs(a.hi - a.lo))

    def test_exp_log_inverse(self):
        x = Ball(1.234, 1e-9)
        y = x.exp().log()
        self.assertLessEqual(y.lo, 1.234)
        self.assertLessEqual(1.234, y.hi)
        self.assertLess(abs(y.lo - 1.234), 1e-6)
        self.assertLess(abs(y.hi - 1.234), 1e-6)

    def test_transcendental_encloses_libm_value(self):
        """The library-returned enclosure must contain the correct-rounded
        value of math.sin(mid) etc. computed independently by the platform."""
        for fn, x in (("sin", 0.3), ("cos", 0.7), ("exp", 0.5),
                      ("log", 2.0), ("atan", 0.9)):
            with self.subTest(fn=fn):
                out = getattr(Ball(x, 1e-9), fn)()
                val = getattr(math, fn)(x)
                self.assertLessEqual(out.lo, val)
                self.assertLessEqual(val, out.hi)

    def test_tan_pole_raises(self):
        with self.assertRaises(ValueError):
            Ball(1.5707963267948966, 1e-1).tan()


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


@unittest.skipUnless(_module_available("mpmath"), "mpmath not installed")
class TestMpmathOracle(unittest.TestCase):
    """Strict 50-digit oracle checks; active when mpmath is installed."""

    @staticmethod
    def _mp():
        try:
            import mpmath
        except ImportError:
            return None
        mpmath.mp.dps = 50
        return mpmath

    def test_transcendental_mpmath_oracle(self):
        mp = self._mp()
        if mp is None:
            self.skipTest("mpmath not installed")
        m = mp.mpf
        for fn, x in (("sin", 0.3), ("cos", 0.7), ("exp", 0.5),
                      ("log", 2.0), ("sqrt", 2.0), ("atan", 0.9),
                      ("sinh", 0.8), ("cosh", 0.8)):
            with self.subTest(fn=fn):
                truth = getattr(mp, fn)(m(x))
                out = getattr(Ball(x, 1e-12), fn)()
                self.assertLessEqual(_dec(out.lo), _dec(str(truth)))
                self.assertLessEqual(_dec(str(truth)), _dec(out.hi))


class TestSemantics(unittest.TestCase):
    def test_overlaps_and_contains(self):
        a = Ball(0.0, 1.0)
        self.assertTrue(a.contains(0.5))
        self.assertIn(0, a)
        self.assertIn(0.0, a)
        self.assertIn(Ball(0.5, 0.0), a)
        self.assertTrue(a.overlaps(Ball(1.2, 0.3)))
        self.assertFalse(a.overlaps(Ball(2.0, 0.1)))

    def test_comparisons_are_interval_wide(self):
        a = Ball(0.0, 0.1)
        self.assertGreater(Ball(1.0, 0.1), a)
        self.assertLess(a, Ball(1.0, 0.1))
        self.assertFalse(a < Ball(0.05, 0.01))

    def test_hash_consistent_with_eq(self):
        a = Ball(Fraction(1, 2), 0.0)
        b = Ball(0.5, 0.0)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_abs(self):
        self.assertEqual(abs(Ball(Fraction(-5))).mid, 5)
        self.assertLessEqual(abs(Ball(-0.5, 1.0)).lo, 0.0)
        self.assertTrue(abs(Ball(-0.5, 1.0)).contains(0.0))


@unittest.skipUnless(_module_available("numpy"), "numpy not installed")
class TestBallArray(unittest.TestCase):
    def test_add_contains(self):
        import numpy as np

        from pyball import BallArray

        arr = BallArray(np.array([1.0, 2.0]), np.array([0.1, 0.1]))
        self.assertEqual(arr.shape, (2,))
        s = arr + arr
        self.assertTrue(s.contains(np.array([2.0, 4.0])).all())
        self.assertIn("BallArray", repr(arr))


class TestVerify(unittest.TestCase):
    def test_prove_positive(self):
        f = lambda b: b ** 2 - 2  # noqa: E731
        self.assertEqual(prove_positive_negative(f, 2.0, 3.0), "positive")

    def test_prove_negative(self):
        g = lambda b: 2 - b * b  # noqa: E731
        self.assertEqual(prove_positive_negative(g, 2.0, 3.0), "negative")

    def test_crossing_zero_inconclusive(self):
        h = lambda b: b - 0.5  # noqa: E731
        self.assertEqual(prove_positive_negative(h, 0.0, 1.0), "inconclusive")

    def test_certify_claim_pred(self):
        from pyball import certify_claim

        self.assertIsNotNone(certify_claim(lambda b: b ** 3 - 1, 0.0, 0.5, "<0"))
        self.assertIsNone(certify_claim(lambda b: b - 0.5, 0.0, 1.0, ">0"))

    def test_certify_claim_enclosure(self):
        from pyball import certify_claim

        e = certify_claim(lambda b: b * b + 1, 0.0, 1.0)
        self.assertIsNotNone(e)
        self.assertGreater(e.lo, 0.0)

    def test_nonfinite_domain_raises(self):
        with self.assertRaises(ValueError):
            prove_positive_negative(lambda b: b, float("inf"), 1.0)
        with self.assertRaises(ValueError):
            prove_positive_negative(lambda b: b, 0.0, float("nan"))


if __name__ == "__main__":
    unittest.main()