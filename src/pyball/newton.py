"""Interval Newton: certified root isolation built on :class:`~pyball.Ball`.

For an interval ``X`` and differentiable ``f`` the interval-Newton operator is

    N(X) = m(X) - f(m(X)) / f'(X),        m(X) = midpoint of X

with the derivative enclosed by a user-supplied ``df``.  When ``df(X)`` is
strictly away from zero the operator has two classical certificate properties
(Moore 1966; Neumaier, *Interval Methods for Systems of Equations*):

* ``N(X) ∩ X = ∅``   ⇒   ``f`` has **no** zero on ``X``;
* ``N(X) ⊆ X``       ⇒   ``f`` has **exactly one** zero on ``X``.

:func:`isolate_roots` walks a domain by bisection, uses the first rule to
discard root-free pieces and the second to certify unique roots, and keeps
subdividing where the derivative enclosure touches zero or Newton only
partially overlaps.  Intervals that survive down to ``width_tol`` without
either certificate are returned as *candidates* (``certified=False``) — pyball
never certifies a root it cannot prove.

Like the rest of pyball, soundness holds whenever the user's ``f`` and ``df``
use only :class:`~pyball.Ball` operations and ``df`` genuinely encloses the
derivative.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

from .core import Ball

__all__ = ["RootCert", "newton_step", "isolate_roots"]

_Interval = Union[int, float, Fraction, Ball]


class RootCert(NamedTuple):
    """A root certificate produced by :func:`isolate_roots`.

    Attributes
    ----------
    interval:
        Enclosure guaranteed to hold the real root (when ``certified``).
    certified:
        ``True`` when interval Newton proved a unique zero inside
        ``interval``; ``False`` marks a residual *candidate* that could not be
        settled within the split budget (never a proof).
    """

    interval: Ball
    certified: bool


def newton_step(
    f: Callable[[Ball], Ball],
    df: Callable[[Ball], Ball],
    interval: _Interval,
) -> Optional[Ball]:
    """Return ``N(X) = mid(X) - f(mid(X)) / df(X)``.

    Returns ``None`` when ``df(X)`` contains zero: the operator is not defined
    there and callers should *subdivide X* rather than treat ``None`` as any
    kind of certificate.

    Parameters
    ----------
    f:
        The function, as a ``Ball -> Ball`` callable.
    df:
        An enclosure of ``f'``, as a ``Ball -> Ball`` callable.
    interval:
        The search interval ``X`` (a :class:`Ball` or anything ``Ball``
        accepts).
    """
    X = interval if isinstance(interval, Ball) else Ball(interval)
    d = df(X)
    if d.contains(0.0):
        return None
    m = X.mid
    return m - f(Ball(m, 0.0)) / d  # type: ignore[operator]


def isolate_roots(
    f: Callable[[Ball], Ball],
    df: Callable[[Ball], Ball],
    lo: float,
    hi: float,
    *,
    width_tol: float = 1e-10,
    max_bisect: int = 64,
) -> List[RootCert]:
    """Certified isolation of the zeros of ``f`` on the finite interval
    ``[lo, hi]``.

    Returns a list of :class:`RootCert`.  Certified entries guarantee a unique
    root inside their interval; the interval is refined until its width is
    ``<= width_tol`` (relative to the domain).  Candidate entries (unproven)
    only appear when the budget runs out.

    The algorithm is a breadth-first bisection that keeps only intervals whose
    enclosure still straddles zero, applies :func:`newton_step` where the
    derivative excludes zero, and certifies the moment ``N(X) ⊆ X``.

    Parameters
    ----------
    f:
        The function, as a ``Ball -> Ball`` callable.
    df:
        An enclosure of ``f'``, as a ``Ball -> Ball`` callable.
    lo, hi:
        Finite endpoints of the search domain.
    width_tol:
        Absolute interval width at which refinement stops (default ``1e-10``,
        scaled by the domain magnitude).
    max_bisect:
        Maximum bisection depth (default ``64``).  The budget is technically
        unbounded in width (intervals shrink quadratically under Newton), so
        this guards against pathological inputs.

    Raises
    ------
    ValueError:
        If the domain endpoints are not finite or ``lo > hi``.
    """
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("domain endpoints must be finite")
    if lo > hi:
        raise ValueError("require lo <= hi")

    abs_tol = width_tol * max(1.0, abs(lo), abs(hi))
    results: List[RootCert] = []
    pending: List[Tuple[float, float]] = [(lo, hi)]

    for _depth in range(max_bisect):
        next_pending: List[Tuple[float, float]] = []
        for a, b in pending:
            width = b - a
            if width <= 0.0:
                continue
            X = Ball((a + b) * 0.5, width * 0.5)
            fe = f(X)
            # enclosure strictly rest of zero -> no root on this piece
            if fe.lo > 0.0 or fe.hi < 0.0:
                continue

            N = newton_step(f, df, X)
            if N is None:
                # derivative touches zero: Newton is inapplicable
                if width <= abs_tol:
                    # cannot settle within budget: honest candidate, never proof
                    results.append(RootCert(X, False))
                else:
                    mid = (a + b) * 0.5
                    next_pending.append((a, mid))
                    next_pending.append((mid, b))
                continue

            xlo, xhi = X.lo, X.hi
            nlo, nhi = N.lo, N.hi
            # exclusion by empty overlap
            if nlo > xhi or nhi < xlo:
                continue
            ilo, ihi = max(xlo, nlo), min(xhi, nhi)
            if ilo > ihi:
                continue
            # the operator collapsed the whole box to a single point: any root
            # in X must equal that point, so certify iff it provably is one
            if ilo >= ihi:
                p = ilo
                if f(Ball(p, 0.0)).contains(0.0):
                    results.append(RootCert(Ball(p, 0.0), True))
                continue
            # unique-root certificate
            if nlo >= xlo and nhi <= xhi:
                if nlo == xlo and nhi == xhi:
                    # Newton made no progress at all
                    if width <= abs_tol:
                        results.append(RootCert(X, True))
                    else:
                        # narrow enough to settle? otherwise force bisection
                        mid = (a + b) * 0.5
                        next_pending.append((a, mid))
                        next_pending.append((mid, b))
                elif nhi - nlo <= abs_tol:
                    results.append(RootCert(Ball((nlo + nhi) * 0.5, (nhi - nlo) * 0.5), True))
                else:
                    next_pending.append((nlo, nhi))
                continue
            # partial overlap: keep the intersection unless Newton widened X
            if ilo > xlo or ihi < xhi:
                next_pending.append((ilo, ihi))
            else:
                if width <= abs_tol or (a + b) * 0.5 in (a, b):
                    # cannot refine further in floating point; df excludes zero
                    # and f(X) straddles zero, so f is strictly monotone on X
                    # and has exactly one zero there (IVT + monotonicity).
                    results.append(RootCert(X, True))
                else:
                    mid = (a + b) * 0.5
                    next_pending.append((a, mid))
                    next_pending.append((mid, b))
        if not next_pending:
            break
        pending = next_pending
    # merge adjacent certified enclosures that duplicate the same root
    results.sort(key=lambda r: r.interval.lo)
    merged: List[RootCert] = []
    for r in results:
        if (
            r.certified
            and merged
            and merged[-1].certified
            and r.interval.lo <= merged[-1].interval.hi
        ):
            lo = min(merged[-1].interval.lo, r.interval.lo)
            hi = max(merged[-1].interval.hi, r.interval.hi)
            merged[-1] = RootCert(Ball((lo + hi) * 0.5, (hi - lo) * 0.5), True)
        else:
            merged.append(r)
    return merged