"""Preprocessing and instance construction utilities."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .model import Option, PricingInstance
from .primitive_admissibility import price_is_admissible


def _as_array_1d(x: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return arr


def from_pricing_data(
    reference_prices: Sequence[float],
    weights: Sequence[float],
    price_menus: Sequence[Sequence[float]],
    demands: Sequence[Sequence[float]],
    uncertainties: Sequence[Sequence[float]],
    margin_target: float,
    tolerances: Sequence[float],
    gamma: int,
    unit_costs: Optional[Sequence[float]] = None,
) -> PricingInstance:
    """Construct a PricingInstance from raw pricing inputs.

    The per-option values are computed as:
        v[i,j] = ω_i * x_{i,j} * ĝ_i(x_{i,j})
        s[i,j] = ω_i * (x_{i,j} - κ q_i) * ĝ_i(x_{i,j})
        t[i,j] = ω_i * |x_{i,j} - κ q_i| * d_adv[i,j]

    Admissible positive prices satisfy |p-r_i| <= σ_i*r_i, evaluated on
    the exact rational values represented by the input floats. Thus ``reference_prices`` control
    admissibility only, while ``unit_costs`` control the economic ratio.

    For releases made before the reference-price/unit-cost distinction was
    explicit, omitting ``unit_costs`` invokes the legacy specialization
    ``q_i = r_i`` and emits a :class:`DeprecationWarning`.  That path preserves
    the historical option coefficients (including their signed uncertainty
    field); all robust solvers use its absolute magnitude.  New code should
    always supply ``unit_costs``.

    Args:
        reference_prices: Admissibility anchors r_i, shape (n,).
        weights: ω_i array, shape (n,).
        price_menus: list of price arrays, per item.
        demands: list of nominal demand arrays, per item.
        uncertainties: Adverse demand-width arrays d_adv, per item.
        margin_target: Revenue-to-cost target κ.
        tolerances: σ_i array, shape (n,).
        gamma: budget Γ.
        unit_costs: Economic unit costs or expected claims q_i, shape (n,).
            Omit only to reproduce the documented legacy case q_i = r_i.

    Returns:
        PricingInstance with filtered options.
    """

    r = _as_array_1d(reference_prices, "reference_prices")
    w = _as_array_1d(weights, "weights")
    sigma = _as_array_1d(tolerances, "tolerances")
    legacy_reference_as_cost = unit_costs is None
    if legacy_reference_as_cost:
        warnings.warn(
            "Omitting unit_costs invokes the legacy q_i = r_i specialization; "
            "pass unit_costs explicitly for the Paper-A v3 model.",
            DeprecationWarning,
            stacklevel=2,
        )
        q = r.copy()
    else:
        q = _as_array_1d(unit_costs, "unit_costs")

    n = r.size
    if w.size != n or sigma.size != n or q.size != n:
        raise ValueError(
            "reference_prices, unit_costs, weights, and tolerances must have same length"
        )
    if len(price_menus) != n or len(demands) != n or len(uncertainties) != n:
        raise ValueError("price_menus, demands, uncertainties must have length n")
    if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("reference_prices must be finite and strictly positive")
    if not np.all(np.isfinite(q)) or np.any(q <= 0.0):
        raise ValueError("unit_costs must be finite and strictly positive")
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
        raise ValueError("tolerances must be finite and nonnegative")
    if not np.isfinite(float(margin_target)) or float(margin_target) <= 0.0:
        raise ValueError("margin_target must be finite and strictly positive")

    items: List[List[Option]] = []

    for i in range(n):
        x = np.asarray(price_menus[i], dtype=float)
        g_hat = np.asarray(demands[i], dtype=float)
        delta = np.asarray(uncertainties[i], dtype=float)
        if x.shape != g_hat.shape or x.shape != delta.shape:
            raise ValueError(f"price, demand, uncertainty shapes mismatch for item {i}")
        if x.ndim != 1:
            raise ValueError(f"price menu for item {i} must be 1D")

        mask = np.asarray([price_is_admissible(p, r[i], sigma[i]) for p in x], dtype=bool)
        if not np.any(mask):
            raise ValueError(f"no admissible options for item {i}")

        x_f = x[mask]
        g_f = g_hat[mask]
        d_f = delta[mask]

        v = w[i] * x_f * g_f
        unit_surplus = x_f - margin_target * q[i]
        margin = w[i] * unit_surplus * g_f
        if legacy_reference_as_cost:
            uncertainty = w[i] * unit_surplus * d_f
        else:
            uncertainty = w[i] * np.abs(unit_surplus) * d_f

        group = [
            Option(value=float(vj), margin=float(sj), uncertainty=float(tj), price=float(xj),
                   original_index=int(original_j))
            for vj, sj, tj, xj, original_j in zip(v, margin, uncertainty, x_f, np.flatnonzero(mask))
        ]
        items.append(group)

    return PricingInstance(items=items, gamma=int(gamma))


def filter_admissible_options(
    instance: PricingInstance,
    reference_prices: Sequence[float],
    tolerances: Sequence[float],
) -> PricingInstance:
    """Filter options to the admissible price interval for low-level instances.

    This helper is useful when constructing PricingInstance directly while still
    enforcing the admissible menu constraint
        x_i ∈ [(1-σ_i) a_i, (1+σ_i) a_i].

    Args:
        instance: PricingInstance with Option.price populated.
        reference_prices: a_i array, shape (n,).
        tolerances: σ_i array, shape (n,).

    Returns:
        A new PricingInstance with inadmissible options removed.
    """

    a = _as_array_1d(reference_prices, "reference_prices")
    sigma = _as_array_1d(tolerances, "tolerances")
    if a.size != instance.n_items or sigma.size != instance.n_items:
        raise ValueError("reference_prices and tolerances must match number of items")

    filtered_items: List[List[Option]] = []
    for i, group in enumerate(instance.items):
        admissible: List[Option] = []
        for j, opt in enumerate(group):
            if opt.price is None:
                raise ValueError("Option.price must be set to filter admissible options")
            if price_is_admissible(opt.price, a[i], sigma[i]):
                admissible.append(opt if opt.original_index is not None else replace(opt, original_index=j))
        if not admissible:
            raise ValueError(f"no admissible options for item {i}")
        filtered_items.append(admissible)

    return PricingInstance(items=filtered_items, gamma=instance.gamma, name=instance.name)
