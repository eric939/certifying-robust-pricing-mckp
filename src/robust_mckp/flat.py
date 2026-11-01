"""Flat binary64 view of an instance, with an exact rational re-check.

The fast solvers work on contiguous arrays; this module builds them from a
:class:`~robust_mckp.model.PricingInstance` or from exact primitive pricing
data, keeps the exact coefficients, and maps selections back to the original
menu IDs.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Union

import numpy as np

from .model import PricingInstance
from .primitive_certificate import ExactPrimitivePricingData


def _exact(x) -> Fraction:
    """The exact rational value of a Fraction, integer or binary64 float."""
    return x if isinstance(x, Fraction) else Fraction(x)


class FlatInstance:
    """Contiguous solver arrays plus the exact coefficients they round.

    Group ``i`` owns the flat options ``off[i] .. off[i+1]-1`` with revenue
    ``v``, nominal surplus ``s`` and endpoint loss ``t >= 0``; each is the
    correctly rounded binary64 value of the exact coefficient.  Legacy
    instances with a negative ``Option.uncertainty`` use ``t = |uncertainty|``
    as everywhere else in the package.
    """

    def __init__(self, instance: Union[PricingInstance, ExactPrimitivePricingData]):
        self.primitive: Optional[ExactPrimitivePricingData] = None
        if isinstance(instance, ExactPrimitivePricingData):
            self.primitive = instance
            instance = instance.encoded_instance()
        if not isinstance(instance, PricingInstance):
            raise TypeError("expected a PricingInstance or ExactPrimitivePricingData")
        self.instance = instance
        self.gamma = int(instance.gamma)
        n = len(instance.items)
        off = np.zeros(n + 1, np.int64)
        v: List[float] = []
        s: List[float] = []
        t: List[float] = []
        self._exact: List[tuple] = []
        for i, group in enumerate(instance.items):
            off[i + 1] = off[i] + len(group)
            for opt in group:
                ev, es, et = _exact(opt.value), _exact(opt.margin), abs(_exact(opt.uncertainty))
                self._exact.append((ev, es, et))
                v.append(float(ev))
                s.append(float(es))
                t.append(float(et))
        self.off = off
        self.v = np.asarray(v, dtype=float)
        self.s = np.asarray(s, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.n = n
        self.M = int(off[-1])

    def thresholds(self) -> np.ndarray:
        """The threshold family B: zero and every distinct endpoint loss."""
        return np.unique(np.concatenate([[0.0], self.t]))

    def local(self, sel: Sequence[int]) -> List[int]:
        """Flat option indices to an index within each group's menu."""
        return [int(k) - int(self.off[i]) for i, k in enumerate(sel)]

    def flat(self, local: Sequence[int]) -> np.ndarray:
        """Indices within each group's menu to flat option indices."""
        if len(local) != self.n:
            raise ValueError(f"selection has {len(local)} entries, expected one per group ({self.n})")
        out = np.empty(self.n, np.int64)
        for i, j in enumerate(local):
            size = int(self.off[i + 1] - self.off[i])
            if not 0 <= int(j) < size:
                raise ValueError(f"group {i}: option {j} outside its menu of {size} options")
            out[i] = int(self.off[i]) + int(j)
        return out

    def original_ids(self, local: Sequence[int]) -> List[int]:
        """Indices within each filtered menu to the IDs of the supplied menus."""
        return self.instance.original_selection([int(j) for j in local])

    def exact_check(self, local: Sequence[int]) -> dict:
        """Exact rational robust residual and revenue of a selection.

        For primitive data the admissibility of every chosen original price is
        re-checked as well.  Returns ``feasible``, ``residual`` and ``revenue``
        (Fractions) and, when infeasible for a structural reason, ``error``.
        """
        flat = self.flat(local)
        residual = Fraction(0)
        revenue = Fraction(0)
        losses = []
        for i, k in enumerate(flat):
            ev, es, et = self._exact[int(k)]
            if self.primitive is not None:
                opt = self.instance.items[i][int(k) - int(self.off[i])]
                d = self.primitive
                p, r, sig = d.menus[i][opt.original_index], d.references[i], d.tolerances[i]
                if abs(p - r) > sig * r:
                    return dict(feasible=False, residual=None, revenue=None,
                                error=f"group {i}: inadmissible original price")
            residual += es
            revenue += ev
            losses.append(et)
        losses.sort(reverse=True)
        residual -= sum(losses[: self.gamma], Fraction(0))
        return dict(feasible=residual >= 0, residual=residual, revenue=revenue)

    def menu_statistics(self) -> dict:
        """Delta^v_max, the lower revenue V_lower and their ratio (Corollary 15(a))."""
        dmax = 0.0
        vlow = 0.0
        for i in range(self.n):
            vi = self.v[self.off[i]:self.off[i + 1]]
            dmax = max(dmax, float(vi.max() - vi.min()))
            vlow += float(vi.min())
        return dict(delta_v_max=dmax, v_lower=vlow, menu_ratio=dmax / vlow if vlow > 0 else None)
