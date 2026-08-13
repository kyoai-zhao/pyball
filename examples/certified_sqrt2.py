"""Example: certified enclosure and prediction interval for a tiny expression.

Run:  PYTHONPATH=src python3 examples/certified_sqrt2.py
"""

from pyball import Ball, certify_claim
from fractions import Fraction

# 1) sqrt(2) is enclosed by Ball(2) ** Fraction(1, 2)
root2 = Ball(Fraction(2)) ** Fraction(1, 2)
print(f"sqrt(2) in {root2}")
assert root2.contains(1.4142135623730951)

# 2) certify that x^3 - 3x + 1 has a sign change on [0, 1] (root in (0,1))
f = lambda b: b**3 - 3 * b + 1
e = certify_claim(f, 0.0, 1.0)
lo, hi = e.lo, e.hi
print(f"f([0,1]) encloses {e}  (must straddle zero: {lo} < 0 < {hi})")
assert lo < 0.0 < hi

# 3) certify a strict inequality: exp(x) > 1.5 on [0.5, 1]
#    (note: exp(x) > 1 + x^2 would NOT be provable on [0,1] because the two
#     sides are equal at x = 0 -- certificates never lie about boundary cases)
g = lambda b: b.exp() - 1.5
from pyball import prove_positive_negative

verdict = prove_positive_negative(g, 0.5, 1.0)
assert verdict == "positive", "exp(x) > 1.5 on [0.5,1] should be provable"
print(f"certified exp(x) > 1.5 on [0.5,1]: verdict={verdict}")

# 4) interval semantics keep comparisons honest
print(f"1.5 in Ball(0,3)? {Ball(1.5, 0.0) in Ball(0.0, 3.0)}")
print("done")