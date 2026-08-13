"""Certified verification helpers built on pyball.

These turn interval enclosures into *verifiable certificates*:

* :func:`prove_positive_negative` — given ``f`` (a python function of a Ball)
  and a search domain, numerically enclose ``f`` on the domain and report
  whether the enclosure proves strict positivity/negativity everywhere.
* :func:`certify_claim` — check a numeric claim of the form
  ``f(x) ? 0`` (or an expression equality) with a rigorous enclosure.

The proofs are machine-checkable only up to the soundness of pyball itself
and of the user's ``f`` (which must use only :class:`~pyball.Ball` ops).
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from .core import Ball, INF

__all__ = ["certify_claim", "prove_positive_negative"]


def _enclosure(
    f: Callable[[Ball], Ball],
    lo: float,
    hi: float,
) -> Ball:
    if not (lo <= hi):
        raise ValueError("require lo <= hi")
    return f(Ball((lo + hi) * 0.5, (hi - lo) * 0.5))


def prove_positive_negative(
    f: Callable[[Ball], Ball],
    lo: float,
    hi: float,
    *,
    max_bisect: int = 40,
) -> str:
    """Return one of ``"positive"``, ``"negative"``, ``"inconclusive"``.

    Bisects ``[lo, hi]``; any sub-interval whose enclosure stays wholly above
    zero is certified positive and adds its length to ``proven`` (likewise for
    negative).  As soon as the proven length covers the whole domain the sign
    is returned.  ``inconclusive`` is returned when the split budget runs out
    or an enclosure straddles zero and cannot be refined.
    """
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("domain endpoints must be finite")

    pending = [(lo, hi)]
    pos_len = neg_len = 0.0
    total = hi - lo
    tol = 1e-10 * max(1.0, abs(hi))
    for _depth in range(max_bisect):
        next_pending = []
        for a, b in pending:
            e = _enclosure(f, a, b)
            if e.lo > 0.0:
                pos_len += b - a
            elif e.hi < 0.0:
                neg_len += b - a
            elif (b - a) < tol:
                return "inconclusive"
            else:
                midp = (a + b) * 0.5
                next_pending.append((a, midp))
                next_pending.append((midp, b))
        if pos_len >= total - tol:
            return "positive"
        if neg_len >= total - tol:
            return "negative"
        if not next_pending:
            return "inconclusive"
        pending = next_pending
    return "inconclusive"


def certify_claim(
    f: Callable[[Ball], Ball],
    lo: float,
    hi: float,
    pred: Optional[str] = None,
) -> Optional[Ball]:
    """Return the enclosure of ``f`` on ``[lo, hi]``.

    If ``pred`` is one of ``">0"`` / ``"<0"`` it additionally runs
    :func:`prove_positive_negative` and only returns the enclosure when that
    sign is provably obtained everywhere on the domain; otherwise ``None``.
    """
    e = _enclosure(f, lo, hi)
    if pred == ">0":
        return e if prove_positive_negative(f, lo, hi) == "positive" else None
    if pred == "<0":
        return e if prove_positive_negative(f, lo, hi) == "negative" else None
    return e