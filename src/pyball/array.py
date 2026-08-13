"""NumPy-vectorized rigorous ball arithmetic.

``BallArray`` stores a pair of ``float64`` arrays (midpoints, radii) and
applies the same directed-rounding enclosures as the scalar :class:`Ball`,
using ``numpy.nextafter`` for outward rounding.  NumPy's arithmetic ufuncs
for ``+ - * / sqrt`` are correctly rounded IEEE-754 operations, so the
``0.5 ulp`` error bound applies exactly as in the scalar core.

Transcendental elementwise functions reuse :meth:`Ball` on a scalar fallback
so that the correctness argument stays single-source (documented in the
README); array users needing hot transcendentals can profile later.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from .core import Ball, _to_float

__all__ = ["BallArray"]

_Float = Union[int, float, np.float64]


def _half_ulp(np_arr: np.ndarray) -> np.ndarray:
    """Elementwise tight upper bound on 0.5 ulp for correctly rounded ops."""
    out = np.zeros_like(np_arr)
    nz = np_arr != 0.0
    out[nz] = (np.nextafter(np_arr[nz], np.inf) - np_arr[nz]) * 0.5
    out[~nz] = 2.5e-322  # half of the smallest subnormal, safe (>= true value)
    return out


class BallArray:
    """An array of ``[mid +/- rad]`` enclosures with rigorous vectorized ops."""

    __slots__ = ("_m", "_r")

    def __init__(self, mid: np.ndarray, rad: np.ndarray | None = None):
        m = np.asarray(mid, dtype=np.float64)
        if rad is None:
            r = np.zeros_like(m)
        else:
            r = np.asarray(rad, dtype=np.float64)
            if r.shape != m.shape:
                raise ValueError(f"mid and rad shapes differ: {m.shape} vs {r.shape}")
            if np.any(r < 0):
                raise ValueError("rad must be non-negative")
        self._m = m
        self._r = r

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_balls(cls, balls) -> "BallArray":
        vals = list(balls)
        if not vals:
            raise ValueError("empty input")
        return cls(
            np.array([b.to_float() for b in vals]),
            np.array([b.rad for b in vals]),
        )

    # -- accessors --------------------------------------------------------
    @property
    def mid(self) -> np.ndarray:
        return self._m

    @property
    def rad(self) -> np.ndarray:
        return self._r

    @property
    def lo(self) -> np.ndarray:
        return np.nextafter(self._m - self._r, -np.inf)

    @property
    def hi(self) -> np.ndarray:
        return np.nextafter(self._m + self._r, np.inf)

    @property
    def shape(self) -> tuple:
        return self._m.shape

    def __len__(self) -> int:
        return len(self._m)

    # -- helpers ----------------------------------------------------------
    def _coerce(self, other) -> "BallArray":
        if isinstance(other, BallArray):
            return other
        m = np.asarray(other, dtype=np.float64)
        return BallArray(m)

    def _from_scalar_result(self, m: np.ndarray, rad: np.ndarray) -> "BallArray":
        return BallArray(m, rad)

    def _result(self, m: np.ndarray, rad: np.ndarray) -> "BallArray":
        return BallArray(m, rad)

    # -- arithmetic -------------------------------------------------------
    def __add__(self, other) -> "BallArray":
        o = self._coerce(other)
        m = self._m + o._m
        err = _half_ulp(m)
        r = np.nextafter(self._r + np.nextafter(o._r + err, np.inf), np.inf)
        return self._result(m, np.where(np.isnan(m), self._r, r))

    def __radd__(self, other) -> "BallArray":
        return self._coerce(other).__add__(self)

    def __sub__(self, other) -> "BallArray":
        o = self._coerce(other)
        m = self._m - o._m
        err = _half_ulp(m)
        r = np.nextafter(self._r + np.nextafter(o._r + err, np.inf), np.inf)
        return self._result(m, np.where(np.isnan(m), self._r, r))

    def __rsub__(self, other) -> "BallArray":
        return self._coerce(other).__sub__(self)

    def __mul__(self, other) -> "BallArray":
        o = self._coerce(other)
        m = self._m * o._m
        err = _half_ulp(m)
        cross = np.abs(self._m) * o._r + np.abs(o._m) * self._r + self._r * o._r + err
        return self._result(m, np.nextafter(cross, np.inf))

    def __rmul__(self, other) -> "BallArray":
        return self._coerce(other).__mul__(self)

    def __truediv__(self, other) -> "BallArray":
        o = self._coerce(other)
        if np.any((o.lo <= 0.0) & (o.hi >= 0.0)):
            raise ZeroDivisionError("division by an element containing zero")
        # elementwise interval division via endpoints
        a1, a2 = self.lo, self.hi
        b1, b2 = o.lo, o.hi
        lo = np.minimum.reduce(
            [np.nextafter(a1 / b2, -np.inf), np.nextafter(a1 / b1, -np.inf),
             np.nextafter(a2 / b2, -np.inf), np.nextafter(a2 / b1, -np.inf)]
        )
        hi = np.maximum.reduce(
            [np.nextafter(a1 / b2, np.inf), np.nextafter(a1 / b1, np.inf),
             np.nextafter(a2 / b2, np.inf), np.nextafter(a2 / b1, np.inf)]
        )
        mid = np.nextafter((lo + hi) * 0.5, np.inf)
        rad = np.nextafter((hi - lo) * 0.5, np.inf)
        return self._result(mid, rad)

    def __rtruediv__(self, other) -> "BallArray":
        return self._coerce(other).__truediv__(self)

    def __neg__(self) -> "BallArray":
        return self._result(-self._m, self._r)

    def __abs__(self) -> "BallArray":
        lo, hi = self.lo, self.hi
        inside = (lo <= 0.0) & (0.0 <= hi)
        mid = np.where(self._m >= 0, self._m, np.where(hi <= 0.0, -self._m, 0.0))
        rad = np.where(inside, np.maximum(np.abs(lo), np.abs(hi)), self._r)
        return self._result(mid, np.nextafter(np.abs(rad), np.inf))

    # -- comparisons ------------------------------------------------------
    def __lt__(self, other) -> np.ndarray:
        o = self._coerce(other)
        return self.hi < o.lo

    def __le__(self, other) -> np.ndarray:
        o = self._coerce(other)
        return self.hi <= o.lo

    def __gt__(self, other) -> np.ndarray:
        o = self._coerce(other)
        return self.lo > o.hi

    def __ge__(self, other) -> np.ndarray:
        o = self._coerce(other)
        return self.lo >= o.hi

    def contains(self, x: np.ndarray) -> np.ndarray:
        return (self.lo <= np.asarray(x, dtype=np.float64)) & (
            np.asarray(x, dtype=np.float64) <= self.hi
        )

    # -- elementary (scalar fallback for correctness) ----------------------
    def exp(self) -> "BallArray":
        return self._map(lambda b: b.exp())

    def log(self) -> "BallArray":
        return self._map(lambda b: b.log())

    def sqrt(self) -> "BallArray":
        return self._map(lambda b: b.sqrt())

    def sin(self) -> "BallArray":
        return self._map(lambda b: b.sin())

    def cos(self) -> "BallArray":
        return self._map(lambda b: b.cos())

    def atan(self) -> "BallArray":
        return self._map(lambda b: b.atan())

    def _map(self, fn) -> "BallArray":
        out_m = np.empty(self._m.shape, dtype=np.float64)
        out_r = np.empty(self._m.shape, dtype=np.float64)
        for idx in np.ndindex(self._m.shape):
            b = fn(Ball(float(self._m[idx]), float(self._r[idx])))
            out_m[idx] = b.to_float()
            out_r[idx] = b.rad
        return self._result(out_m, out_r)

    def __repr__(self) -> str:
        lo, hi = self.lo, self.hi
        flat = [(float(a), float(b)) for a, b in zip(
            np.ravel(lo), np.ravel(hi)
        )][:4]
        more = "..." if flat and np.prod(self.shape) > 4 else ""
        return f"BallArray(shape={self.shape}, first=[{', '.join(f'[{a:.6g}, {b:.6g}]' for a, b in flat)}]{more})"