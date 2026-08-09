"""Data models for robust MCKP pricing optimization."""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class Option:
    """An option within a pricing menu.

    Attributes:
        value: Objective coefficient v[i,j].
        margin: Nominal margin contribution s[i,j].
        uncertainty: Uncertainty contribution t[i,j].
        price: Optional price level x_{i,j} for reference.
    """

    value: float
    margin: float
    uncertainty: float
    price: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("value", "margin", "uncertainty"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.price is not None:
            if not isinstance(self.price, numbers.Real) or isinstance(self.price, bool):
                raise TypeError("price must be a real number when provided")
            if not math.isfinite(float(self.price)):
                raise ValueError("price must be finite when provided")


@dataclass
class PricingInstance:
    """Robust pricing instance for the multiple-choice knapsack structure.

    Attributes:
        items: List of item groups, each a list of Option objects.
        gamma: Budget parameter Γ (integer in [0, n]).
        name: Optional label for logging/reporting.
    """

    items: List[List[Option]]
    gamma: int
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.gamma, numbers.Integral) or isinstance(self.gamma, bool):
            raise TypeError("gamma must be an integer")
        if self.gamma < 0:
            raise ValueError("gamma must be nonnegative")
        if not self.items:
            raise ValueError("items must be non-empty")
        for group in self.items:
            if not group:
                raise ValueError("each item group must be non-empty")
        if self.gamma > len(self.items):
            raise ValueError("gamma must be at most the number of items")

    @property
    def n_items(self) -> int:
        """Number of item groups."""

        return len(self.items)

    @property
    def n_options(self) -> int:
        """Total number of options across all groups."""

        return sum(len(group) for group in self.items)


@dataclass
class Solution:
    """Solution container for the robust MCKP solver.

    Attributes:
        selections: Selected option index for each item (list of length n).
        objective: Objective value Σ v[i, j(i)].
        lp_value: Optimal LP relaxation value at the chosen θ.
        gap_to_lp: Relative gap (lp_value - objective) / lp_value.
        certificate_value: Robust margin certificate Z = Σ s_i - β.
        theta: θ associated with the chosen candidate.
        elapsed: Wall-clock time in seconds.
        is_feasible: Whether the returned solution satisfies the robust constraint.
        metadata: Optional extra info for debugging.
    """

    selections: List[int]
    objective: float
    lp_value: float
    gap_to_lp: float
    certificate_value: float
    theta: float
    elapsed: float
    is_feasible: bool
    metadata: Optional[dict] = field(default=None)
