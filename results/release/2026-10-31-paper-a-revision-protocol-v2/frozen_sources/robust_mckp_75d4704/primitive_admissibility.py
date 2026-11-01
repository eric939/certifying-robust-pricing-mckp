"""Strict pricing admissibility for the exact values represented by input floats."""
from fractions import Fraction
import math


def price_is_admissible(price: float, reference: float, tolerance: float) -> bool:
    """Check |p-r| <= sigma*r without rounding its sides or adding EPS."""
    p, r, sigma = float(price), float(reference), float(tolerance)
    if not math.isfinite(r) or r <= 0 or not math.isfinite(sigma) or sigma < 0:
        raise ValueError('Finite positive reference and finite nonnegative tolerance required')
    if not math.isfinite(p) or p <= 0:
        return False
    return abs(Fraction.from_float(p)-Fraction.from_float(r)) <= (
        Fraction.from_float(sigma)*Fraction.from_float(r))


def inward_generated_price(price: float, reference: float, tolerance: float) -> float:
    """Move a generated boundary price inward by at most eight float steps.

    Supplied menus use strict filtering. This helper is for generated prices;
    it rejects candidates requiring more than eight steps and preserves every
    already admissible price.
    """
    candidate = float(price)
    if not math.isfinite(candidate) or candidate <= 0:
        raise ValueError('Generated price must be finite and positive')
    for step in range(9):
        if price_is_admissible(candidate, reference, tolerance):
            return candidate
        if step < 8:
            candidate = math.nextafter(candidate, float(reference))
    raise ValueError('Generated price is outside the admissible interval by more than eight float steps')
