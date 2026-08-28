"""Exact, independently checkable global certificates for robust MCKP.

The optimization algorithms in :mod:`robust_mckp` use floating-point LP
bounds and numerical comparison tolerances.  This module is deliberately
separate from those algorithms.  It interprets every input coefficient as the
exact rational number represented by its binary64 bits, retains the complete
set consisting of zero and every distinct original deviation, and records one
exact Lagrangian multiplier per feasible fixed-threshold MCKP.

For a fixed threshold ``theta``, write

    q[i,j] = s[i,j] - max(0, |t[i,j]| - theta),
    h[i]   = max_j q[i,j],
    c[i,j] = h[i] - q[i,j],
    C      = sum_i h[i] - Gamma * theta.

When ``C >= 0``, every nonnegative multiplier ``lambda`` supplies the valid LP
dual bound

    U_theta(lambda) = lambda*C + sum_i max_j(v[i,j] - lambda*c[i,j]).

The sharp global LP upper bound is the maximum of these fixed-threshold dual
minima.  The checker below recomputes all transformed coefficients and group
maxima from the original instance, and it verifies the exact subgradient
optimality condition for every serialized multiplier.  Thus neither derived
coefficients nor a merely feasible but nonoptimal dual evaluation are accepted
as the sharp bound ``U``.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from .model import PricingInstance


CERTIFICATE_KIND = "robust_mckp_global_dual_certificate"
CERTIFICATE_SCHEMA_VERSION = 1

FractionInput = Union[Fraction, int, float]


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _parse_fraction(value: Any, field: str) -> Fraction:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a canonical rational string")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid rational in {field}: {value!r}") from exc
    if _fraction_text(parsed) != value:
        raise ValueError(f"{field} is not in canonical rational form")
    return parsed


def _as_exact_fraction(value: FractionInput, field: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{field} must be a real number or Fraction")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    binary64 = float(value)
    if not math.isfinite(binary64):
        raise ValueError(f"{field} must be finite")
    return Fraction.from_float(binary64)


def _binary64_fraction(value: numbers.Real) -> Fraction:
    """Return the exact rational represented by ``float(value)``."""

    binary64 = float(value)
    if not math.isfinite(binary64):
        raise ValueError("instance coefficients must be finite")
    return Fraction.from_float(binary64)


def instance_sha256(instance: PricingInstance) -> str:
    """Hash the ordered binary64 instance coefficients and group structure."""

    groups = []
    for group in instance.items:
        encoded_group = []
        for option in group:
            encoded_group.append(
                {
                    "value": float(option.value).hex(),
                    "margin": float(option.margin).hex(),
                    "uncertainty": float(option.uncertainty).hex(),
                    "price": None if option.price is None else float(option.price).hex(),
                }
            )
        groups.append(encoded_group)
    payload = {
        "coefficient_interpretation": "exact_binary64",
        "gamma": int(instance.gamma),
        "groups": groups,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def original_theta_candidates(instance: PricingInstance) -> Tuple[Fraction, ...]:
    """Return zero and every original binary64-distinct absolute deviation."""

    candidates = {Fraction(0)}
    for group in instance.items:
        for option in group:
            candidates.add(abs(_binary64_fraction(option.uncertainty)))
    return tuple(sorted(candidates))


@dataclass(frozen=True)
class FixedThetaDualCertificate:
    """One exact multiplier and bound for a fixed threshold."""

    theta: Fraction
    feasible: bool
    multiplier: Optional[Fraction] = None
    upper_bound: Optional[Fraction] = None

    def __post_init__(self) -> None:
        if self.theta < 0:
            raise ValueError("theta must be nonnegative")
        if self.feasible:
            if self.multiplier is None or self.upper_bound is None:
                raise ValueError("a feasible threshold requires multiplier and upper bound")
            if self.multiplier < 0:
                raise ValueError("dual multiplier must be nonnegative")
        elif self.multiplier is not None or self.upper_bound is not None:
            raise ValueError("an infeasible threshold must not carry a multiplier or bound")

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "theta": _fraction_text(self.theta),
            "feasible": self.feasible,
        }
        if self.feasible:
            assert self.multiplier is not None and self.upper_bound is not None
            payload["lambda"] = _fraction_text(self.multiplier)
            payload["U_theta"] = _fraction_text(self.upper_bound)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FixedThetaDualCertificate":
        if not isinstance(payload, Mapping):
            raise TypeError("threshold certificate must be an object")
        expected = {"theta", "feasible"}
        feasible = payload.get("feasible")
        if not isinstance(feasible, bool):
            raise TypeError("threshold feasible flag must be boolean")
        if feasible:
            expected.update({"lambda", "U_theta"})
        if set(payload) != expected:
            raise ValueError("threshold certificate has missing or unexpected fields")
        return cls(
            theta=_parse_fraction(payload["theta"], "theta"),
            feasible=feasible,
            multiplier=(
                _parse_fraction(payload["lambda"], "lambda") if feasible else None
            ),
            upper_bound=(
                _parse_fraction(payload["U_theta"], "U_theta") if feasible else None
            ),
        )


@dataclass(frozen=True)
class GlobalDualCertificate:
    """Compact global incumbent and fixed-threshold LP-dual certificate."""

    instance_hash: str
    selections: Tuple[int, ...]
    h: Fraction
    u: Fraction
    robust_residual: Fraction
    absolute_gap: Fraction
    relative_width: Optional[Fraction]
    thresholds: Tuple[FixedThetaDualCertificate, ...]
    schema_version: int = CERTIFICATE_SCHEMA_VERSION
    kind: str = CERTIFICATE_KIND

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "instance_sha256": self.instance_hash,
            "selections": list(self.selections),
            "H": _fraction_text(self.h),
            "U": _fraction_text(self.u),
            "robust_residual": _fraction_text(self.robust_residual),
            "absolute_gap": _fraction_text(self.absolute_gap),
            "relative_width": (
                None
                if self.relative_width is None
                else _fraction_text(self.relative_width)
            ),
            "thresholds": [record.to_dict() for record in self.thresholds],
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlobalDualCertificate":
        if not isinstance(payload, Mapping):
            raise TypeError("global certificate must be an object")
        expected = {
            "schema_version",
            "kind",
            "instance_sha256",
            "selections",
            "H",
            "U",
            "robust_residual",
            "absolute_gap",
            "relative_width",
            "thresholds",
        }
        if set(payload) != expected:
            raise ValueError("global certificate has missing or unexpected fields")
        if payload["schema_version"] != CERTIFICATE_SCHEMA_VERSION:
            raise ValueError("unsupported global certificate schema version")
        if payload["kind"] != CERTIFICATE_KIND:
            raise ValueError("unexpected certificate kind")
        instance_hash = payload["instance_sha256"]
        if not isinstance(instance_hash, str) or len(instance_hash) != 64:
            raise ValueError("instance_sha256 must be a 64-character string")
        raw_selections = payload["selections"]
        if not isinstance(raw_selections, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_selections
        ):
            raise TypeError("selections must be a list of integers")
        raw_thresholds = payload["thresholds"]
        if not isinstance(raw_thresholds, list):
            raise TypeError("thresholds must be a list")
        raw_relative_width = payload["relative_width"]
        return cls(
            instance_hash=instance_hash,
            selections=tuple(raw_selections),
            h=_parse_fraction(payload["H"], "H"),
            u=_parse_fraction(payload["U"], "U"),
            robust_residual=_parse_fraction(
                payload["robust_residual"], "robust_residual"
            ),
            absolute_gap=_parse_fraction(payload["absolute_gap"], "absolute_gap"),
            relative_width=(
                None
                if raw_relative_width is None
                else _parse_fraction(raw_relative_width, "relative_width")
            ),
            thresholds=tuple(
                FixedThetaDualCertificate.from_dict(record)
                for record in raw_thresholds
            ),
        )

    @classmethod
    def from_json(cls, raw: str) -> "GlobalDualCertificate":
        payload = json.loads(raw)
        return cls.from_dict(payload)


@dataclass(frozen=True)
class GlobalCertificateVerification:
    """Result returned by the independent certificate checker."""

    valid: bool
    errors: Tuple[str, ...]
    exact_h: Optional[Fraction] = None
    exact_u: Optional[Fraction] = None
    exact_robust_residual: Optional[Fraction] = None
    exact_absolute_gap: Optional[Fraction] = None
    exact_relative_width: Optional[Fraction] = None


ExactGroups = Tuple[Tuple[Fraction, ...], ...]


def _recompute_fixed_theta(
    instance: PricingInstance, theta: Fraction
) -> Tuple[ExactGroups, ExactGroups, Fraction]:
    """Recompute values, transformed costs, and capacity from original data."""

    values = []
    costs = []
    stars = []
    for group in instance.items:
        group_values = tuple(_binary64_fraction(option.value) for option in group)
        transformed = tuple(
            _binary64_fraction(option.margin)
            - max(
                Fraction(0),
                abs(_binary64_fraction(option.uncertainty)) - theta,
            )
            for option in group
        )
        star = max(transformed)
        values.append(group_values)
        costs.append(tuple(star - value for value in transformed))
        stars.append(star)
    capacity = sum(stars, Fraction(0)) - int(instance.gamma) * theta
    return tuple(values), tuple(costs), capacity


def _dual_upper_bound(
    values: ExactGroups,
    costs: ExactGroups,
    capacity: Fraction,
    multiplier: Fraction,
) -> Fraction:
    if multiplier < 0:
        raise ValueError("dual multiplier must be nonnegative")
    return multiplier * capacity + sum(
        (max(v - multiplier * c for v, c in zip(group_values, group_costs))
         for group_values, group_costs in zip(values, costs)),
        Fraction(0),
    )


def _right_derivative(
    values: ExactGroups,
    costs: ExactGroups,
    capacity: Fraction,
    multiplier: Fraction,
) -> Fraction:
    """Right derivative of the convex piecewise-linear dual function."""

    active_min_cost_sum = Fraction(0)
    for group_values, group_costs in zip(values, costs):
        scores = tuple(
            value - multiplier * cost
            for value, cost in zip(group_values, group_costs)
        )
        best = max(scores)
        active_min_cost_sum += min(
            cost for score, cost in zip(scores, group_costs) if score == best
        )
    return capacity - active_min_cost_sum


def _left_derivative(
    values: ExactGroups,
    costs: ExactGroups,
    capacity: Fraction,
    multiplier: Fraction,
) -> Fraction:
    """Left derivative of the convex dual function at a positive multiplier."""

    active_max_cost_sum = Fraction(0)
    for group_values, group_costs in zip(values, costs):
        scores = tuple(
            value - multiplier * cost
            for value, cost in zip(group_values, group_costs)
        )
        best = max(scores)
        active_max_cost_sum += max(
            cost for score, cost in zip(scores, group_costs) if score == best
        )
    return capacity - active_max_cost_sum


def _dual_multiplier_is_optimal(
    values: ExactGroups,
    costs: ExactGroups,
    capacity: Fraction,
    multiplier: Fraction,
) -> bool:
    """Check exact optimality over ``lambda >= 0`` by subgradients."""

    if multiplier < 0:
        return False
    right = _right_derivative(values, costs, capacity, multiplier)
    if multiplier == 0:
        return right >= 0
    left = _left_derivative(values, costs, capacity, multiplier)
    return left <= 0 <= right


def _exact_dual_minimizer(
    values: ExactGroups, costs: ExactGroups, capacity: Fraction
) -> Fraction:
    """Choose an exact minimizer of ``U_theta`` over nonnegative multipliers."""

    breakpoints = {Fraction(0)}
    for group_values, group_costs in zip(values, costs):
        size = len(group_values)
        for left in range(size):
            for right in range(left + 1, size):
                cost_difference = group_costs[left] - group_costs[right]
                if cost_difference == 0:
                    continue
                crossing = (
                    group_values[left] - group_values[right]
                ) / cost_difference
                if crossing >= 0:
                    breakpoints.add(crossing)

    ordered = sorted(breakpoints)
    if _right_derivative(values, costs, capacity, ordered[0]) >= 0:
        return ordered[0]

    low = 1
    high = len(ordered) - 1
    while low < high:
        middle = (low + high) // 2
        if _right_derivative(values, costs, capacity, ordered[middle]) >= 0:
            high = middle
        else:
            low = middle + 1

    candidates = [ordered[low]]
    if low > 0:
        candidates.append(ordered[low - 1])
    return min(
        candidates,
        key=lambda multiplier: (
            _dual_upper_bound(values, costs, capacity, multiplier),
            multiplier,
        ),
    )


def _exact_objective(instance: PricingInstance, selections: Sequence[int]) -> Fraction:
    return sum(
        (
            _binary64_fraction(instance.items[item][int(option)].value)
            for item, option in enumerate(selections)
        ),
        Fraction(0),
    )


def _exact_robust_residual(
    instance: PricingInstance, selections: Sequence[int]
) -> Fraction:
    margins = []
    deviations = []
    for item, option_index in enumerate(selections):
        option = instance.items[item][int(option_index)]
        margins.append(_binary64_fraction(option.margin))
        deviations.append(abs(_binary64_fraction(option.uncertainty)))
    deviations.sort(reverse=True)
    gamma = min(max(int(instance.gamma), 0), len(deviations))
    return sum(margins, Fraction(0)) - sum(
        deviations[:gamma], Fraction(0)
    )


def _validate_selections(instance: PricingInstance, selections: Sequence[int]) -> Tuple[int, ...]:
    if len(selections) != instance.n_items:
        raise ValueError("selection length must match the number of groups")
    normalized = []
    for item, option in enumerate(selections):
        if isinstance(option, bool) or not isinstance(option, numbers.Integral):
            raise TypeError("selection indices must be integers")
        index = int(option)
        if index < 0 or index >= len(instance.items[item]):
            raise ValueError(f"selection index out of range for group {item}")
        normalized.append(index)
    return tuple(normalized)


def _relative_width(u: Fraction, absolute_gap: Fraction) -> Optional[Fraction]:
    """Return ``(U-H)/U`` when the upper endpoint is positive."""

    if u > 0:
        return absolute_gap / u
    return None


def _normalize_multiplier_map(
    multipliers: Mapping[FractionInput, FractionInput],
) -> Dict[Fraction, Fraction]:
    normalized: Dict[Fraction, Fraction] = {}
    for raw_theta, raw_multiplier in multipliers.items():
        theta = _as_exact_fraction(raw_theta, "multiplier theta")
        multiplier = _as_exact_fraction(raw_multiplier, "multiplier")
        if theta in normalized:
            raise ValueError("duplicate exact theta in multiplier mapping")
        if multiplier < 0:
            raise ValueError("dual multiplier must be nonnegative")
        normalized[theta] = multiplier
    return normalized


def build_global_certificate(
    instance: PricingInstance,
    selections: Sequence[int],
    *,
    multipliers: Optional[Mapping[FractionInput, FractionInput]] = None,
) -> GlobalDualCertificate:
    """Build an exact global LP-dual gap certificate for a robust incumbent.

    If ``multipliers`` is omitted, an exact minimizer of each one-dimensional
    fixed-threshold dual is selected.  A supplied mapping must contain exactly
    one *optimal* nonnegative multiplier for every feasible original threshold
    and no entries for infeasible thresholds.  Exact subgradient conditions
    are checked before a supplied multiplier is recorded as defining ``U``.
    """

    normalized_selections = _validate_selections(instance, selections)
    robust_residual = _exact_robust_residual(instance, normalized_selections)
    if robust_residual < 0:
        raise ValueError("the supplied incumbent is not robust feasible")

    supplied = None if multipliers is None else _normalize_multiplier_map(multipliers)
    records = []
    feasible_thetas = set()
    for theta in original_theta_candidates(instance):
        values, costs, capacity = _recompute_fixed_theta(instance, theta)
        if capacity < 0:
            records.append(FixedThetaDualCertificate(theta=theta, feasible=False))
            continue
        feasible_thetas.add(theta)
        if supplied is None:
            multiplier = _exact_dual_minimizer(values, costs, capacity)
        else:
            if theta not in supplied:
                raise ValueError(f"missing multiplier for feasible theta {_fraction_text(theta)}")
            multiplier = supplied[theta]
        if not _dual_multiplier_is_optimal(values, costs, capacity, multiplier):
            raise ValueError(
                "multiplier is not dual optimal at theta "
                f"{_fraction_text(theta)}"
            )
        upper_bound = _dual_upper_bound(values, costs, capacity, multiplier)
        records.append(
            FixedThetaDualCertificate(
                theta=theta,
                feasible=True,
                multiplier=multiplier,
                upper_bound=upper_bound,
            )
        )

    if supplied is not None and set(supplied) != feasible_thetas:
        raise ValueError("multiplier mapping contains an infeasible or unknown theta")

    feasible_bounds = [
        record.upper_bound
        for record in records
        if record.feasible and record.upper_bound is not None
    ]
    if not feasible_bounds:
        raise ValueError("no feasible fixed threshold exists")
    h = _exact_objective(instance, normalized_selections)
    u = max(feasible_bounds)
    absolute_gap = u - h
    if absolute_gap < 0:
        raise AssertionError("a recomputed LP dual bound lies below the feasible incumbent")
    return GlobalDualCertificate(
        instance_hash=instance_sha256(instance),
        selections=normalized_selections,
        h=h,
        u=u,
        robust_residual=robust_residual,
        absolute_gap=absolute_gap,
        relative_width=_relative_width(u, absolute_gap),
        thresholds=tuple(records),
    )


def verify_global_certificate(
    instance: PricingInstance,
    certificate: Union[GlobalDualCertificate, Mapping[str, Any], str],
) -> GlobalCertificateVerification:
    """Independently recompute and verify a serialized global certificate.

    The checker trusts only the original instance, the incumbent indices, and
    the serialized thresholds/multipliers.  It reconstructs every cost,
    capacity, group maximum, fixed-threshold bound, and aggregate width exactly.
    It also proves each multiplier minimizes its one-dimensional dual through
    the exact left/right derivative condition, so an accepted ``U`` is the
    sharp full-family LP bound rather than a generic dual-witness bound.
    """

    try:
        if isinstance(certificate, GlobalDualCertificate):
            parsed = GlobalDualCertificate.from_dict(certificate.to_dict())
        elif isinstance(certificate, str):
            parsed = GlobalDualCertificate.from_json(certificate)
        else:
            parsed = GlobalDualCertificate.from_dict(certificate)
    except Exception as exc:
        return GlobalCertificateVerification(
            valid=False,
            errors=(f"malformed certificate: {exc}",),
        )

    errors = []
    if parsed.instance_hash != instance_sha256(instance):
        errors.append("instance hash mismatch")

    try:
        selections = _validate_selections(instance, parsed.selections)
    except Exception as exc:
        return GlobalCertificateVerification(
            valid=False,
            errors=tuple(errors + [f"invalid selections: {exc}"]),
        )

    exact_h = _exact_objective(instance, selections)
    exact_residual = _exact_robust_residual(instance, selections)
    if exact_residual < 0:
        errors.append("incumbent fails the original exact sorted-Gamma constraint")
    if parsed.h != exact_h:
        errors.append("stored H does not equal the exact incumbent objective")
    if parsed.robust_residual != exact_residual:
        errors.append("stored robust residual does not match exact recomputation")

    expected_thetas = original_theta_candidates(instance)
    observed_thetas = tuple(record.theta for record in parsed.thresholds)
    if observed_thetas != expected_thetas:
        errors.append("threshold list is not the complete ordered original family")

    recomputed_bounds = []
    if len(parsed.thresholds) == len(expected_thetas):
        for expected_theta, record in zip(expected_thetas, parsed.thresholds):
            if record.theta != expected_theta:
                continue
            # Reconstruct from original option coefficients.  In particular,
            # no stored capacity, group maximum, cost, or bound is consumed.
            values, costs, capacity = _recompute_fixed_theta(instance, expected_theta)
            expected_feasible = capacity >= 0
            if record.feasible != expected_feasible:
                errors.append(
                    f"incorrect feasibility flag at theta {_fraction_text(expected_theta)}"
                )
                continue
            if not expected_feasible:
                continue
            assert record.multiplier is not None and record.upper_bound is not None
            if record.multiplier < 0:
                errors.append(
                    f"negative multiplier at theta {_fraction_text(expected_theta)}"
                )
                continue
            exact_bound = _dual_upper_bound(
                values, costs, capacity, record.multiplier
            )
            recomputed_bounds.append(exact_bound)
            if not _dual_multiplier_is_optimal(
                values, costs, capacity, record.multiplier
            ):
                errors.append(
                    f"nonoptimal multiplier at theta {_fraction_text(expected_theta)}"
                )
            if record.upper_bound != exact_bound:
                errors.append(
                    f"stored U_theta mismatch at theta {_fraction_text(expected_theta)}"
                )

    exact_u: Optional[Fraction] = None
    exact_absolute_gap: Optional[Fraction] = None
    exact_relative_width: Optional[Fraction] = None
    if recomputed_bounds:
        exact_u = max(recomputed_bounds)
        exact_absolute_gap = exact_u - exact_h
        exact_relative_width = _relative_width(exact_u, exact_absolute_gap)
        if exact_absolute_gap < 0:
            errors.append("global dual upper bound is below the feasible incumbent")
        if parsed.u != exact_u:
            errors.append("stored U does not match recomputed threshold bounds")
        if parsed.absolute_gap != exact_absolute_gap:
            errors.append("stored absolute gap does not match exact U-H")
        if parsed.relative_width != exact_relative_width:
            errors.append("stored relative width does not match exact recomputation")
    else:
        errors.append("certificate contains no recomputable feasible threshold bound")

    return GlobalCertificateVerification(
        valid=not errors,
        errors=tuple(errors),
        exact_h=exact_h,
        exact_u=exact_u,
        exact_robust_residual=exact_residual,
        exact_absolute_gap=exact_absolute_gap,
        exact_relative_width=exact_relative_width,
    )


__all__ = [
    "CERTIFICATE_KIND",
    "CERTIFICATE_SCHEMA_VERSION",
    "FixedThetaDualCertificate",
    "GlobalCertificateVerification",
    "GlobalDualCertificate",
    "build_global_certificate",
    "instance_sha256",
    "original_theta_candidates",
    "verify_global_certificate",
]
