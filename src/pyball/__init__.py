"""pyball: rigorous ball/interval arithmetic for Python.

A small, auditable library providing verified enclosures ``[mid ± rad]``
with directed rounding, exact rational fast paths, elementary functions,
an optional NumPy-backed vectorized layer, and simple certified-evaluation
helpers.

Public API:
    Ball       -- scalar rigorous enclosure
    BallArray  -- vectorized enclosures (requires ``numpy``)
    INF, NaN   -- module-level IEEE-754 constants
"""

from .core import INF, NaN, Ball
from .verify import certify_claim, prove_positive_negative

try:  # numpy is an optional extra
    from .array import BallArray
except ImportError:  # pragma: no cover - exercised when numpy is absent
    BallArray = None  # type: ignore[assignment,misc]

__version__ = "0.1.0"

__all__ = [
    "Ball",
    "BallArray",
    "INF",
    "NaN",
    "certify_claim",
    "prove_positive_negative",
    "__version__",
]