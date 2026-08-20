"""pyball: rigorous ball/interval arithmetic for Python.

A ``Ball`` represents the enclosure ``[mid - rad, mid + rad]`` of a real
number.  All operations return an enclosure that is guaranteed to contain
every result obtainable from the input enclosures, *including* the rounding
errors of the machine arithmetic used to compute it.

Rigor model
-----------
* Midpoint may be an exact :class:`fractions.Fraction` (or an exact rational
  like ``int``).  When the midpoint is exact and the radius is ``0.0``,
  arithmetic between two such balls stays exact (no rounding is involved).
* When float arithmetic is used, every result endpoint is rounded *outward*
  with ``math.nextafter`` (directed rounding), and rounding error is covered
  by the standard correctly-rounded bound ``0.5 * ulp`` for the IEEE-754
  basic operations that Python's ``float`` guarantees.
* Transcendental functions use a Lipschitz bound on the whole input ball to
  inflate the radius:  ``f(m +/- r) <= f(m) + (L*r + 0.5*ulp(f(m)))``.
  The bounding requires ``f`` to be defined and Lipschitz on the input ball.

This is sound but intentionally simple: it favours a small, auditable core
over the aggressive tightness of Arb's Taylor/Richardson machinery.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Union

__all__ = ["Ball", "INF", "NaN"]

INF = math.inf
NaN = math.nan

# A midpoint is either an exact rational or a machine float.
_Exact = Union[int, Fraction]
_Mid = Union[_Exact, float]

#: value of one ulp at 0.0 (smallest positive subnormal): covers the
#: rounding of operations that land on exactly zero.
_MIN_SUBNORMAL = 5e-324
_HALF_ULP_ZERO = _MIN_SUBNORMAL / 2


def _up(x: float) -> float:
    """Next float >= x (directed rounding up)."""
    return math.nextafter(x, INF)


def _down(x: float) -> float:
    """Next float <= x (directed rounding down)."""
    return math.nextafter(x, -INF)


def _half_ulp(s: float) -> float:
    """Upper bound on the error of a *correctly rounded* op to ``s``.

    IEEE-754 basic operations (+, -, *, /, sqrt) are correctly rounded in
    Python's ``float``, so round-to-nearest error is <= 0.5 ulp(s).  We return
    a value >= 0.5 ulp(s) so the bound holds even if it is slightly loose.
    """
    if s == 0.0:
        return _HALF_ULP_ZERO
    if math.isinf(s) or math.isnan(s):
        return 0.0
    up = math.nextafter(s, INF)
    half = (up - s) * 0.5
    # protect the half-ulp itself from rounding down below 0.5 ulp
    return _up(half)


def _ulp(s: float) -> float:
    """Upper bound on the error of a libm transcendental call returning ``s``.

    ``math.exp/log/sin/...`` are NOT guaranteed correctly rounded by the C
    standard; IEEE-754-2019 recommends they be accurate to 1 ulp.  We bound
    their error by a full ulp, which is sound under that platform assumption
    (documented in the README).  Using a full ulp (instead of the 0.5 ulp
    reserved for basic operations) is the point of this function.
    """
    if s == 0.0:
        return _MIN_SUBNORMAL
    if math.isinf(s) or math.isnan(s):
        return 0.0
    up = math.nextafter(s, INF)
    return _up(up - s)


def _to_float(m: _Mid) -> float:
    """Exact value of an exact rational midpoint as a float (rounded)."""
    if isinstance(m, Fraction):
        return float(m)
    return float(m)


class Ball:
    """A rigorous enclosure ``[mid - rad, mid + rad]``.

    Parameters
    ----------
    mid:
        Midpoint: exact rational (``int``/``Fraction``) or ``float``.
    rad:
        Non-negative radius.  Must be a real ``float``; ``0.0`` means the
        ball represents exactly ``mid`` (provided ``mid`` is exact).
    """

    __slots__ = ("_m", "_r")

    def __init__(self, mid: Union[_Mid, "Ball"], rad: float = 0.0):
        if isinstance(mid, Ball):
            if rad != 0.0:
                raise ValueError("cannot pass a Ball and a nonzero rad")
            self._m = mid._m
            self._r = mid._r
            return
        if isinstance(mid, bool):
            raise TypeError("mid must be a number, not bool")
        if not isinstance(mid, (int, float, Fraction)):
            raise TypeError(f"unsupported midpoint type: {type(mid).__name__}")
        if isinstance(mid, float) and math.isnan(mid):
            raise ValueError("NaN midpoint is not allowed")
        if not isinstance(rad, (int, float)) or isinstance(rad, bool):
            raise TypeError(f"rad must be a real number, not {type(rad).__name__}")
        rad = float(rad)
        if rad < 0.0:
            raise ValueError(f"rad must be >= 0, got {rad!r}")
        if math.isnan(rad):
            raise ValueError("NaN radius is not allowed")
        self._m = mid if isinstance(mid, Fraction) else (Fraction(mid) if isinstance(mid, int) else mid)
        self._r = rad

    # -- accessors -------------------------------------------------------
    @property
    def mid(self) -> _Mid:
        return self._m

    @property
    def rad(self) -> float:
        return self._r

    @property
    def lo(self) -> float:
        """Rounded-down lower endpoint."""
        return _down(_to_float(self._m) - self._r)

    @property
    def hi(self) -> float:
        """Rounded-up upper endpoint."""
        return _up(_to_float(self._m) + self._r)

    @property
    def is_exact(self) -> bool:
        return isinstance(self._m, Fraction) and self._r == 0.0

    def to_float(self) -> float:
        """The midpoint as a plain float (rounded; loses exactness info)."""
        return _to_float(self._m)

    def contains(self, x: Union[int, float, Fraction]) -> bool:
        """True if ``x`` is provably inside the enclosure."""
        return self.lo <= float(x) <= self.hi

    def overlaps(self, other: "Ball") -> bool:
        return self.lo <= other.hi and other.lo <= self.hi

    def __contains__(self, x: object) -> bool:
        if isinstance(x, Ball):
            return x.lo >= self.lo and x.hi <= self.hi
        return isinstance(x, (int, float, Fraction)) and self.contains(x)  # type: ignore[arg-type]

    # -- equality / ordering (interval semantics) -------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ball):
            try:
                other = Ball(other)
            except (TypeError, ValueError):
                return NotImplemented
        return self._m == other._m and self._r == other._r

    def __hash__(self) -> int:
        return hash((self._m, self._r))

    def __lt__(self, other: Union[_Mid, "Ball"]) -> bool:
        """True iff the whole enclosure lies strictly below the other's."""
        o = other if isinstance(other, Ball) else Ball(other)
        return self.hi < o.lo

    def __le__(self, other: Union[_Mid, "Ball"]) -> bool:
        o = other if isinstance(other, Ball) else Ball(other)
        return self.hi <= o.lo

    def __gt__(self, other: Union[_Mid, "Ball"]) -> bool:
        o = other if isinstance(other, Ball) else Ball(other)
        return self.lo > o.hi

    def __ge__(self, other: Union[_Mid, "Ball"]) -> bool:
        o = other if isinstance(other, Ball) else Ball(other)
        return self.lo >= o.hi

    # -- exact helpers ----------------------------------------------------
    @staticmethod
    def _exact_pair(a: "Ball", b: "Ball") -> bool:
        return a.is_exact and b.is_exact

    @classmethod
    def _from_endpoints(cls, lo: float, hi: float) -> "Ball":
        """Build the tightest ball enclosing [lo, hi] given outward-rounded
        endpoints (``lo <= true_lo``, ``hi >= true_hi``)."""
        L, H = Fraction(lo), Fraction(hi)
        mid = (L + H) / 2
        m = float(mid)
        # reach of the chosen float midpoint from each side, kept exact;
        # rounding the radius up guarantees the enclosure never shrinks
        rad = _up(float(max(mid - L, H - mid)))
        return cls(m, rad)

    # -- arithmetic -------------------------------------------------------
    def __neg__(self) -> "Ball":
        return Ball(-self._m, self._r)

    def __pos__(self) -> "Ball":
        return Ball(self)

    def __add__(self, other: Union[_Mid, "Ball"]) -> "Ball":
        o = other if isinstance(other, Ball) else Ball(other)
        if self._exact_pair(self, o):
            return Ball(self._m + o._m, 0.0)
        m = _to_float(self._m) + _to_float(o._m)
        err = _half_ulp(m)
        rad = _up(self._r + _up(o._r + err))
        return Ball(m, rad)

    def __radd__(self, other: Union[_Mid]) -> "Ball":
        return Ball(other).__add__(self)

    def __sub__(self, other: Union[_Mid, "Ball"]) -> "Ball":
        o = other if isinstance(other, Ball) else Ball(other)
        if self._exact_pair(self, o):
            return Ball(self._m - o._m, 0.0)
        m = _to_float(self._m) - _to_float(o._m)
        err = _half_ulp(m)
        rad = _up(self._r + _up(o._r + err))
        return Ball(m, rad)

    def __rsub__(self, other: Union[_Mid]) -> "Ball":
        return Ball(other).__sub__(self)

    def __mul__(self, other: Union[_Mid, "Ball"]) -> "Ball":
        o = other if isinstance(other, Ball) else Ball(other)
        if self._exact_pair(self, o):
            return Ball(self._m * o._m, 0.0)
        m = _to_float(self._m) * _to_float(o._m)
        err = _half_ulp(m)
        # radii combine via |m1|*r2 + |m2|*r1 + r1*r2 (+ rounding of the product)
        cross = abs(self._m) * o._r + abs(o._m) * self._r + self._r * o._r + err
        rad = _up(cross)
        return Ball(m, rad)

    def __rmul__(self, other: Union[_Mid]) -> "Ball":
        return Ball(other).__mul__(self)

    def __truediv__(self, other: Union[_Mid, "Ball"]) -> "Ball":
        o = other if isinstance(other, Ball) else Ball(other)
        if o.contains(0):
            raise ZeroDivisionError("division by a ball containing zero")
        if self._exact_pair(self, o):
            return Ball(self._m / o._m, 0.0)
        # endpoint interval division: A/B = [a1,a2]*[1/b2,1/b1], each endpoint
        # op correctly rounded and then pushed outward.
        a1, a2 = self.lo, self.hi
        b1, b2 = o.lo, o.hi
        lo = min(_down(a1 / b2), _down(a1 / b1), _down(a2 / b2), _down(a2 / b1))
        hi = max(_up(a1 / b2), _up(a1 / b1), _up(a2 / b2), _up(a2 / b1))
        return self._from_endpoints(lo, hi)

    def __rtruediv__(self, other: Union[_Mid]) -> "Ball":
        return Ball(other).__truediv__(self)

    def __pow__(self, exp: Union[int, Fraction, "Ball"]) -> "Ball":
        if isinstance(exp, Ball):
            if exp.is_exact and isinstance(exp._m, Fraction):
                return self.__pow__(exp._m)
            # x**y = exp(y*log(x)); requires positive enclosure
            if self.lo <= 0.0:
                raise ValueError("Ball**Ball requires a positive base")
            return (exp * self.log()).exp()
        if isinstance(exp, bool):
            raise TypeError("exponent must be an int, Fraction, float or Ball")
        if isinstance(exp, float):
            # the machine float denotes the exact rational equal to its binary
            # value (e.g. 0.5 == 1/2, 0.1 == 3602879701896397/2^55); route
            # through the rational-exponent machinery, which is rigorous.
            if math.isnan(exp) or math.isinf(exp):
                raise ValueError("NaN/inf exponent is not supported")
            exp = Fraction(*exp.as_integer_ratio())
        if isinstance(exp, (int, Fraction)):
            if exp == 0:
                return Ball(1)
            if exp == 1:
                return Ball(self)
            neg = exp < 0
            e = -exp if neg else exp
            if self.is_exact and (e == int(e)):
                # exact rational base ^ integer exponent stays exact
                return Ball(self._m ** int(e) if not neg else 1 / (self._m ** int(e)), 0.0)
            if isinstance(e, Fraction) and e.denominator != 1:
                # rational exponent via exp(log) on a positive base
                if self.lo <= 0.0:
                    raise ValueError("negative-base fractional powers are not real-valued")
                result = (e * self.log()).exp()
            else:
                n = int(e)
                result = Ball(1)
                base = Ball(self)
                while n:
                    if n & 1:
                        result = result * base
                    base = base * base
                    n >>= 1
            if neg:
                if self.contains(0):
                    raise ZeroDivisionError("negative power of a ball containing zero")
                result = Ball(1) / result
            return result
        raise TypeError(f"unsupported exponent type: {type(exp).__name__}")

    # -- unary / elementary functions -------------------------------------
    def __abs__(self) -> "Ball":
        if self._m >= 0:
            return Ball(self)
        if self.hi <= 0.0:
            return -self
        # interval crosses zero: [0, max(|lo|,|hi|)]
        return Ball(0.0, _up(max(abs(self.lo), abs(self.hi))))

    def sqrt(self) -> "Ball":
        if self.hi < 0.0:
            raise ValueError("sqrt of a negative enclosure")
        lo = _down(math.sqrt(max(0.0, self.lo)))
        hi = _up(math.sqrt(self.hi))
        return self._from_endpoints(lo, hi)

    def exp(self) -> "Ball":
        # f = exp, df/dx = exp; L = exp(hi), error of exp(m) bounded by 1 ulp
        hi = self.hi
        L = _up(math.exp(hi))
        m = math.exp(_to_float(self._m))
        err = _ulp(m)
        rad = _up(L * self._r + err)
        return Ball(m, rad)

    def log(self) -> "Ball":
        if self.lo <= 0.0:
            raise ValueError("log requires a strictly positive enclosure")
        lo, hi = self.lo, self.hi
        L = _up(1.0 / lo)  # derivative 1/x max at lo
        m = math.log(_to_float(self._m))
        err = _ulp(m)
        rad = _up(L * self._r + err)
        return Ball(m, rad)

    def sin(self) -> "Ball":
        m = math.sin(_to_float(self._m))
        err = _ulp(m)
        # |d/dx sin| = |cos| <= 1
        rad = _up(self._r + err)
        return Ball(m, rad)

    def cos(self) -> "Ball":
        m = math.cos(_to_float(self._m))
        err = _ulp(m)
        rad = _up(self._r + err)
        return Ball(m, rad)

    def tan(self) -> "Ball":
        # need no pole inside the ball: sec^2 = 1 + tan^2 diverges at poles.
        m = math.tan(_to_float(self._m))
        err = _ulp(m)
        c = self.cos()
        if c.contains(0.0):
            raise ValueError("tan undefined near a pole (cos ball contains zero)")
        # |sec^2| = 1/cos^2 <= 1/(min |cos|)^2 over the enclosure; cos is
        # monotone-free but bounded below in magnitude by the endpoint.
        minabs = min(abs(c.lo), abs(c.hi))
        L = _up(1.0 / (minabs * minabs))
        rad = _up(L * self._r + err)
        return Ball(m, rad)

    def atan(self) -> "Ball":
        m = math.atan(_to_float(self._m))
        err = _ulp(m)
        # |d/dx atan| = 1/(1+x^2) <= 1
        rad = _up(self._r + err)
        return Ball(m, rad)

    def sinh(self) -> "Ball":
        a = max(abs(self.lo), abs(self.hi))
        L = _up(math.cosh(a))
        m = math.sinh(_to_float(self._m))
        err = _ulp(m)
        return Ball(m, _up(L * self._r + err))

    def cosh(self) -> "Ball":
        # derivative sinh, |sinh| max at the farthest endpoint
        a = max(abs(self.lo), abs(self.hi))
        L = _up(math.sinh(a))
        m = math.cosh(_to_float(self._m))
        err = _ulp(m)
        return Ball(m, _up(L * self._r + err))

    # -- misc -------------------------------------------------------------
    def __repr__(self) -> str:
        if self.is_exact:
            return f"Ball({self._m!r}, 0.0)"
        return f"Ball({self.to_float()!r}, {self._r!r})"

    def __str__(self) -> str:
        if self.is_exact:
            return f"[{self._m}]"
        return f"[{self.lo!r}, {self.hi!r}]"
