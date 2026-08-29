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
ROUNDING_PROVENANCE_SCHEMA_VERSION = 2

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
class ExactLPGroupPosition:
    """One group's exact position in a basic continuous-MCKP solution.

    Integral groups repeat the selected option and use zero upper weight.
    A fractional group names two distinct adjacent upper-hull options and the
    exact weight on the upper-cost endpoint.
    """

    lower_option: int
    upper_option: int
    upper_weight: Fraction

    def __post_init__(self) -> None:
        for field, value in (
            ("lower_option", self.lower_option),
            ("upper_option", self.upper_option),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
            if value < 0:
                raise ValueError(f"{field} must be nonnegative")
        if not isinstance(self.upper_weight, Fraction):
            raise TypeError("upper_weight must be a Fraction")
        if self.lower_option == self.upper_option:
            if self.upper_weight != 0:
                raise ValueError("an integral group must have zero upper weight")
        elif not 0 < self.upper_weight < 1:
            raise ValueError(
                "a fractional group must have upper weight strictly between zero and one"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower_option": self.lower_option,
            "upper_option": self.upper_option,
            "upper_weight": _fraction_text(self.upper_weight),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExactLPGroupPosition":
        if not isinstance(payload, Mapping):
            raise TypeError("LP group position must be an object")
        expected = {"lower_option", "upper_option", "upper_weight"}
        if set(payload) != expected:
            raise ValueError("LP group position has missing or unexpected fields")
        lower = payload["lower_option"]
        upper = payload["upper_option"]
        if isinstance(lower, bool) or not isinstance(lower, int):
            raise TypeError("lower_option must be an integer")
        if isinstance(upper, bool) or not isinstance(upper, int):
            raise TypeError("upper_option must be an integer")
        return cls(
            lower_option=lower,
            upper_option=upper,
            upper_weight=_parse_fraction(payload["upper_weight"], "upper_weight"),
        )


@dataclass(frozen=True)
class LPToRoundingProvenance:
    """Exact LP-basis and literal round-down witness at a global maximizer."""

    theta: Fraction
    groups: Tuple[ExactLPGroupPosition, ...]
    fractional_group: Optional[int]
    fractional_lower_option: Optional[int]
    fractional_upper_option: Optional[int]
    mixing_fraction: Optional[Fraction]
    rounded_selections: Tuple[int, ...]
    primal_cost: Fraction
    primal_value: Fraction
    rounded_value: Fraction
    rounding_loss: Fraction
    adjacent_value_jump: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.theta, Fraction) or self.theta < 0:
            raise ValueError("provenance theta must be a nonnegative Fraction")
        if not isinstance(self.groups, tuple) or any(
            not isinstance(position, ExactLPGroupPosition)
            for position in self.groups
        ):
            raise TypeError("provenance groups must be exact LP group positions")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.rounded_selections
        ):
            raise TypeError("rounded selections must be nonnegative integers")
        rational_fields = (
            self.primal_cost,
            self.primal_value,
            self.rounded_value,
            self.rounding_loss,
            self.adjacent_value_jump,
        )
        if any(not isinstance(value, Fraction) for value in rational_fields):
            raise TypeError("provenance numerical fields must be Fractions")

        fractional_metadata = (
            self.fractional_group,
            self.fractional_lower_option,
            self.fractional_upper_option,
            self.mixing_fraction,
        )
        if self.fractional_group is None:
            if any(value is not None for value in fractional_metadata[1:]):
                raise ValueError(
                    "integral provenance must have null fractional metadata"
                )
        else:
            for field, value in (
                ("fractional_group", self.fractional_group),
                ("fractional_lower_option", self.fractional_lower_option),
                ("fractional_upper_option", self.fractional_upper_option),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise TypeError(f"{field} must be a nonnegative integer")
            if not isinstance(self.mixing_fraction, Fraction):
                raise TypeError("mixing_fraction must be a Fraction")
            if not 0 < self.mixing_fraction < 1:
                raise ValueError("mixing_fraction must lie strictly between zero and one")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theta": _fraction_text(self.theta),
            "groups": [position.to_dict() for position in self.groups],
            "fractional_group": self.fractional_group,
            "fractional_lower_option": self.fractional_lower_option,
            "fractional_upper_option": self.fractional_upper_option,
            "mixing_fraction": (
                None
                if self.mixing_fraction is None
                else _fraction_text(self.mixing_fraction)
            ),
            "rounded_selections": list(self.rounded_selections),
            "primal_cost": _fraction_text(self.primal_cost),
            "primal_value": _fraction_text(self.primal_value),
            "rounded_value": _fraction_text(self.rounded_value),
            "rounding_loss": _fraction_text(self.rounding_loss),
            "adjacent_value_jump": _fraction_text(self.adjacent_value_jump),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LPToRoundingProvenance":
        if not isinstance(payload, Mapping):
            raise TypeError("rounding provenance must be an object")
        expected = {
            "theta",
            "groups",
            "fractional_group",
            "fractional_lower_option",
            "fractional_upper_option",
            "mixing_fraction",
            "rounded_selections",
            "primal_cost",
            "primal_value",
            "rounded_value",
            "rounding_loss",
            "adjacent_value_jump",
        }
        if set(payload) != expected:
            raise ValueError("rounding provenance has missing or unexpected fields")
        raw_groups = payload["groups"]
        if not isinstance(raw_groups, list):
            raise TypeError("rounding provenance groups must be a list")
        raw_rounded = payload["rounded_selections"]
        if not isinstance(raw_rounded, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_rounded
        ):
            raise TypeError("rounded_selections must be a list of integers")
        fractional_group = payload["fractional_group"]
        fractional_lower = payload["fractional_lower_option"]
        fractional_upper = payload["fractional_upper_option"]
        for field, value in (
            ("fractional_group", fractional_group),
            ("fractional_lower_option", fractional_lower),
            ("fractional_upper_option", fractional_upper),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{field} must be an integer or null")
        raw_mixing = payload["mixing_fraction"]
        return cls(
            theta=_parse_fraction(payload["theta"], "provenance theta"),
            groups=tuple(
                ExactLPGroupPosition.from_dict(position)
                for position in raw_groups
            ),
            fractional_group=fractional_group,
            fractional_lower_option=fractional_lower,
            fractional_upper_option=fractional_upper,
            mixing_fraction=(
                None
                if raw_mixing is None
                else _parse_fraction(raw_mixing, "mixing_fraction")
            ),
            rounded_selections=tuple(raw_rounded),
            primal_cost=_parse_fraction(payload["primal_cost"], "primal_cost"),
            primal_value=_parse_fraction(payload["primal_value"], "primal_value"),
            rounded_value=_parse_fraction(payload["rounded_value"], "rounded_value"),
            rounding_loss=_parse_fraction(payload["rounding_loss"], "rounding_loss"),
            adjacent_value_jump=_parse_fraction(
                payload["adjacent_value_jump"], "adjacent_value_jump"
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
    rounding_provenance: Optional[LPToRoundingProvenance] = None
    schema_version: int = CERTIFICATE_SCHEMA_VERSION
    kind: str = CERTIFICATE_KIND

    def __post_init__(self) -> None:
        if self.schema_version == CERTIFICATE_SCHEMA_VERSION:
            if self.rounding_provenance is not None:
                raise ValueError(
                    "schema version 1 cannot carry LP-to-rounding provenance"
                )
        elif self.schema_version == ROUNDING_PROVENANCE_SCHEMA_VERSION:
            if self.rounding_provenance is None:
                raise ValueError(
                    "schema version 2 requires LP-to-rounding provenance"
                )
        else:
            raise ValueError("unsupported global certificate schema version")

    def to_dict(self) -> Dict[str, Any]:
        payload = {
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
        if self.rounding_provenance is not None:
            payload["rounding_provenance"] = self.rounding_provenance.to_dict()
        return payload

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GlobalDualCertificate":
        if not isinstance(payload, Mapping):
            raise TypeError("global certificate must be an object")
        base_fields = {
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
        schema_version = payload.get("schema_version")
        if schema_version == CERTIFICATE_SCHEMA_VERSION:
            expected = base_fields
        elif schema_version == ROUNDING_PROVENANCE_SCHEMA_VERSION:
            expected = base_fields | {"rounding_provenance"}
        else:
            raise ValueError("unsupported global certificate schema version")
        if set(payload) != expected:
            raise ValueError("global certificate has missing or unexpected fields")
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
            rounding_provenance=(
                None
                if schema_version == CERTIFICATE_SCHEMA_VERSION
                else LPToRoundingProvenance.from_dict(
                    payload["rounding_provenance"]
                )
            ),
            schema_version=schema_version,
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
    rounding_provenance_verified: bool = False
    exact_rounding_loss: Optional[Fraction] = None


ExactGroups = Tuple[Tuple[Fraction, ...], ...]


@dataclass(frozen=True)
class _ExactHullVertex:
    option: int
    cost: Fraction
    value: Fraction


@dataclass(frozen=True)
class _ExactHullSegment:
    group: int
    index: int
    slope: Fraction
    length: Fraction


def _exact_upper_hull(
    group_values: Tuple[Fraction, ...],
    group_costs: Tuple[Fraction, ...],
) -> Tuple[_ExactHullVertex, ...]:
    """Construct the relevant concave upper hull without floating arithmetic."""

    best_at_cost: Dict[Fraction, _ExactHullVertex] = {}
    for option, (cost, value) in enumerate(zip(group_costs, group_values)):
        candidate = _ExactHullVertex(option=option, cost=cost, value=value)
        incumbent = best_at_cost.get(cost)
        if incumbent is None or value > incumbent.value or (
            value == incumbent.value and option < incumbent.option
        ):
            best_at_cost[cost] = candidate

    nondominated = []
    best_value: Optional[Fraction] = None
    for cost in sorted(best_at_cost):
        vertex = best_at_cost[cost]
        if best_value is None or vertex.value > best_value:
            nondominated.append(vertex)
            best_value = vertex.value

    hull = []
    for vertex in nondominated:
        while len(hull) >= 2:
            origin = hull[-2]
            middle = hull[-1]
            cross = (
                (middle.cost - origin.cost) * (vertex.value - origin.value)
                - (middle.value - origin.value) * (vertex.cost - origin.cost)
            )
            if cross < 0:
                break
            hull.pop()
        hull.append(vertex)
    if not hull:
        raise AssertionError("an exact group hull cannot be empty")
    return tuple(hull)


def _build_lp_to_rounding_provenance(
    instance: PricingInstance,
    theta: Fraction,
    expected_primal_value: Fraction,
) -> LPToRoundingProvenance:
    """Build a deterministic exact basic LP solution and its round-down point."""

    values, costs, capacity = _recompute_fixed_theta(instance, theta)
    if capacity < 0:
        raise AssertionError("cannot build LP provenance at an infeasible threshold")
    hulls = tuple(
        _exact_upper_hull(group_values, group_costs)
        for group_values, group_costs in zip(values, costs)
    )

    current_vertices = [0] * instance.n_items
    base_cost = sum((hull[0].cost for hull in hulls), Fraction(0))
    residual = capacity - base_cost
    if residual < 0:
        raise AssertionError("zero-cost baselines should fit every feasible threshold")

    segments = []
    for group, hull in enumerate(hulls):
        for index, (lower, upper) in enumerate(zip(hull, hull[1:])):
            length = upper.cost - lower.cost
            value_jump = upper.value - lower.value
            if length <= 0 or value_jump <= 0:
                raise AssertionError("exact upper-hull segments must increase cost and value")
            segments.append(
                _ExactHullSegment(
                    group=group,
                    index=index,
                    slope=value_jump / length,
                    length=length,
                )
            )
    segments.sort(key=lambda segment: (-segment.slope, segment.group, segment.index))

    fractional_group: Optional[int] = None
    fractional_segment: Optional[int] = None
    mixing_fraction: Optional[Fraction] = None
    for segment in segments:
        if residual == 0:
            break
        if current_vertices[segment.group] != segment.index:
            raise AssertionError("upper-hull segment order violated within a group")
        if residual >= segment.length:
            current_vertices[segment.group] += 1
            residual -= segment.length
            continue
        fractional_group = segment.group
        fractional_segment = segment.index
        mixing_fraction = residual / segment.length
        residual = Fraction(0)
        break

    positions = []
    rounded = []
    primal_cost = Fraction(0)
    primal_value = Fraction(0)
    for group, hull in enumerate(hulls):
        current = current_vertices[group]
        if group == fractional_group:
            assert fractional_segment is not None and mixing_fraction is not None
            lower = hull[fractional_segment]
            upper = hull[fractional_segment + 1]
            positions.append(
                ExactLPGroupPosition(
                    lower_option=lower.option,
                    upper_option=upper.option,
                    upper_weight=mixing_fraction,
                )
            )
            rounded.append(lower.option)
            primal_cost += lower.cost + mixing_fraction * (upper.cost - lower.cost)
            primal_value += lower.value + mixing_fraction * (
                upper.value - lower.value
            )
        else:
            vertex = hull[current]
            positions.append(
                ExactLPGroupPosition(
                    lower_option=vertex.option,
                    upper_option=vertex.option,
                    upper_weight=Fraction(0),
                )
            )
            rounded.append(vertex.option)
            primal_cost += vertex.cost
            primal_value += vertex.value

    if primal_cost > capacity:
        raise AssertionError("constructed exact LP witness exceeds capacity")
    if fractional_group is not None and primal_cost != capacity:
        raise AssertionError("a fractional basic witness must bind capacity")
    if primal_value != expected_primal_value:
        raise AssertionError(
            "exact upper-hull primal does not equal the sharp fixed-threshold dual"
        )

    rounded_selections = tuple(rounded)
    rounded_value = _exact_objective(instance, rounded_selections)
    if _exact_robust_residual(instance, rounded_selections) < 0:
        raise AssertionError("exact fixed-threshold rounding is not robust feasible")
    rounding_loss = primal_value - rounded_value
    if fractional_group is None:
        fractional_lower_option = None
        fractional_upper_option = None
        adjacent_value_jump = Fraction(0)
    else:
        position = positions[fractional_group]
        fractional_lower_option = position.lower_option
        fractional_upper_option = position.upper_option
        adjacent_value_jump = (
            values[fractional_group][position.upper_option]
            - values[fractional_group][position.lower_option]
        )
    return LPToRoundingProvenance(
        theta=theta,
        groups=tuple(positions),
        fractional_group=fractional_group,
        fractional_lower_option=fractional_lower_option,
        fractional_upper_option=fractional_upper_option,
        mixing_fraction=mixing_fraction,
        rounded_selections=rounded_selections,
        primal_cost=primal_cost,
        primal_value=primal_value,
        rounded_value=rounded_value,
        rounding_loss=rounding_loss,
        adjacent_value_jump=adjacent_value_jump,
    )


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
    include_rounding_provenance: bool = False,
) -> GlobalDualCertificate:
    """Build an exact global LP-dual gap certificate for a robust incumbent.

    If ``multipliers`` is omitted, an exact minimizer of each one-dimensional
    fixed-threshold dual is selected.  A supplied mapping must contain exactly
    one *optimal* nonnegative multiplier for every feasible original threshold
    and no entries for infeasible thresholds.  Exact subgradient conditions
    are checked before a supplied multiplier is recorded as defining ``U``.

    Set ``include_rounding_provenance`` to emit a version-2 extension carrying
    an exact basic LP solution at a threshold attaining the global ``U``, its
    literal one-group round-down selection, and the exact loss.  The supplied
    incumbent must dominate that rounded value, as required when it represents
    the lower endpoint ``H`` of the full-family construction.
    """

    if not isinstance(include_rounding_provenance, bool):
        raise TypeError("include_rounding_provenance must be boolean")

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
    rounding_provenance = None
    schema_version = CERTIFICATE_SCHEMA_VERSION
    if include_rounding_provenance:
        maximizing_record = next(
            record
            for record in records
            if record.feasible and record.upper_bound == u
        )
        rounding_provenance = _build_lp_to_rounding_provenance(
            instance,
            maximizing_record.theta,
            u,
        )
        if h < rounding_provenance.rounded_value:
            raise ValueError(
                "the supplied incumbent H does not dominate the literal rounded "
                "selection at the global LP-maximizing threshold"
            )
        schema_version = ROUNDING_PROVENANCE_SCHEMA_VERSION
    return GlobalDualCertificate(
        instance_hash=instance_sha256(instance),
        selections=normalized_selections,
        h=h,
        u=u,
        robust_residual=robust_residual,
        absolute_gap=absolute_gap,
        relative_width=_relative_width(u, absolute_gap),
        thresholds=tuple(records),
        rounding_provenance=rounding_provenance,
        schema_version=schema_version,
    )


def _verify_lp_to_rounding_provenance(
    instance: PricingInstance,
    provenance: LPToRoundingProvenance,
    thresholds: Tuple[FixedThetaDualCertificate, ...],
    exact_h: Fraction,
    exact_u: Optional[Fraction],
) -> Tuple[Tuple[str, ...], Optional[Fraction]]:
    """Verify a provenance block solely from original coefficients and witnesses."""

    errors = []
    matching_records = [
        record for record in thresholds if record.theta == provenance.theta
    ]
    if len(matching_records) != 1:
        errors.append(
            "rounding provenance theta is not unique in the complete threshold family"
        )
        threshold_record = None
    else:
        threshold_record = matching_records[0]

    values, costs, capacity = _recompute_fixed_theta(instance, provenance.theta)
    if capacity < 0:
        errors.append("rounding provenance uses an infeasible threshold")

    if len(provenance.groups) != instance.n_items:
        errors.append("rounding provenance must contain one LP position per group")

    primal_cost = Fraction(0)
    primal_value = Fraction(0)
    fractional_groups = []
    expected_rounded = []
    positions_recomputable = len(provenance.groups) == instance.n_items
    exact_fractional_jump: Optional[Fraction] = None
    exact_fractional_loss: Optional[Fraction] = None

    if positions_recomputable:
        for group, position in enumerate(provenance.groups):
            group_size = len(instance.items[group])
            if (
                position.lower_option >= group_size
                or position.upper_option >= group_size
            ):
                errors.append(
                    f"rounding provenance option index out of range for group {group}"
                )
                positions_recomputable = False
                continue

            lower = position.lower_option
            upper = position.upper_option
            lower_cost = costs[group][lower]
            upper_cost = costs[group][upper]
            lower_value = values[group][lower]
            upper_value = values[group][upper]
            hull = _exact_upper_hull(values[group], costs[group])
            hull_coordinates = tuple((vertex.cost, vertex.value) for vertex in hull)

            if lower == upper:
                if (lower_cost, lower_value) not in hull_coordinates:
                    errors.append(
                        f"integral LP option is not on the exact upper hull in group {group}"
                    )
                expected_rounded.append(lower)
                primal_cost += lower_cost
                primal_value += lower_value
                continue

            fractional_groups.append(group)
            adjacent = any(
                (left.cost, left.value) == (lower_cost, lower_value)
                and (right.cost, right.value) == (upper_cost, upper_value)
                for left, right in zip(hull, hull[1:])
            )
            if not adjacent:
                errors.append(
                    f"fractional endpoints are not an adjacent exact upper-hull edge in group {group}"
                )
            alpha = position.upper_weight
            expected_rounded.append(lower)
            primal_cost += lower_cost + alpha * (upper_cost - lower_cost)
            primal_value += lower_value + alpha * (upper_value - lower_value)
            exact_fractional_jump = upper_value - lower_value
            exact_fractional_loss = alpha * exact_fractional_jump

    if len(fractional_groups) > 1:
        errors.append("exact basic LP representation has more than one fractional group")
    actual_fractional_group = (
        fractional_groups[0] if len(fractional_groups) == 1 else None
    )
    if provenance.fractional_group != actual_fractional_group:
        errors.append("fractional-group metadata does not match the LP positions")
    if actual_fractional_group is None:
        if provenance.adjacent_value_jump != 0:
            errors.append("integral LP provenance must have zero adjacent value jump")
        exact_fractional_jump = Fraction(0)
        exact_fractional_loss = Fraction(0)
    else:
        position = provenance.groups[actual_fractional_group]
        if provenance.fractional_lower_option != position.lower_option:
            errors.append("fractional lower endpoint metadata mismatch")
        if provenance.fractional_upper_option != position.upper_option:
            errors.append("fractional upper endpoint metadata mismatch")
        if provenance.mixing_fraction != position.upper_weight:
            errors.append("mixing fraction metadata does not match the LP position")

    if positions_recomputable:
        if provenance.primal_cost != primal_cost:
            errors.append("stored primal cost does not match exact LP recomputation")
        if provenance.primal_value != primal_value:
            errors.append("stored primal value does not match exact LP recomputation")
        if primal_cost > capacity:
            errors.append("exact LP witness exceeds fixed-threshold capacity")
        if actual_fractional_group is not None and primal_cost != capacity:
            errors.append("fractional basic LP witness does not bind capacity")

    exact_dual_value: Optional[Fraction] = None
    if threshold_record is not None:
        if not threshold_record.feasible:
            errors.append("rounding provenance threshold has no feasible dual record")
        else:
            assert threshold_record.multiplier is not None
            exact_dual_value = _dual_upper_bound(
                values,
                costs,
                capacity,
                threshold_record.multiplier,
            )
            if not _dual_multiplier_is_optimal(
                values,
                costs,
                capacity,
                threshold_record.multiplier,
            ):
                errors.append("rounding provenance is paired with a nonoptimal dual")
            if positions_recomputable and primal_value != exact_dual_value:
                errors.append("exact primal-dual equality fails at provenance theta")
            if exact_u is not None and exact_dual_value != exact_u:
                errors.append(
                    "rounding provenance theta does not attain the global LP maximum"
                )

    exact_rounded_value: Optional[Fraction] = None
    try:
        rounded_selections = _validate_selections(
            instance, provenance.rounded_selections
        )
    except Exception as exc:
        errors.append(f"invalid literal rounded selection: {exc}")
    else:
        if positions_recomputable and rounded_selections != tuple(expected_rounded):
            errors.append(
                "literal rounded selection does not take the lower fractional endpoint"
            )
        rounded_cost = sum(
            (
                costs[group][option]
                for group, option in enumerate(rounded_selections)
            ),
            Fraction(0),
        )
        if rounded_cost > capacity:
            errors.append("literal rounded selection exceeds fixed-threshold capacity")
        rounded_residual = _exact_robust_residual(instance, rounded_selections)
        if rounded_residual < 0:
            errors.append(
                "literal rounded selection fails the original exact sorted-Gamma constraint"
            )
        exact_rounded_value = _exact_objective(instance, rounded_selections)
        if provenance.rounded_value != exact_rounded_value:
            errors.append("stored rounded value does not match exact recomputation")
        if exact_h < exact_rounded_value:
            errors.append("certificate H does not dominate the literal rounded value")

    exact_rounding_loss: Optional[Fraction] = None
    if positions_recomputable and exact_rounded_value is not None:
        exact_rounding_loss = primal_value - exact_rounded_value
        if provenance.rounding_loss != exact_rounding_loss:
            errors.append("stored rounding loss does not match exact recomputation")
        if exact_fractional_loss is not None and (
            exact_rounding_loss != exact_fractional_loss
        ):
            errors.append("rounding loss is not the fractional adjacent-edge loss")
        if exact_fractional_jump is not None:
            if provenance.adjacent_value_jump != exact_fractional_jump:
                errors.append(
                    "stored adjacent value jump does not match exact recomputation"
                )
            if not 0 <= exact_rounding_loss <= exact_fractional_jump:
                errors.append("rounding loss is outside the adjacent-jump bound")
        if exact_u is not None and exact_u - exact_h > exact_rounding_loss:
            errors.append("certificate U-H is not dominated by the rounding loss")

    return tuple(errors), exact_rounding_loss


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
    Version-2 certificates additionally reconstruct and check the exact basic
    primal, fractional hull edge, literal round-down point, primal-dual
    equality at a global LP-maximizing threshold, and domination by ``H``.
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

    exact_rounding_loss: Optional[Fraction] = None
    provenance_errors: Tuple[str, ...] = ()
    if parsed.rounding_provenance is not None:
        provenance_errors, exact_rounding_loss = _verify_lp_to_rounding_provenance(
            instance,
            parsed.rounding_provenance,
            parsed.thresholds,
            exact_h,
            exact_u,
        )
        errors.extend(provenance_errors)

    return GlobalCertificateVerification(
        valid=not errors,
        errors=tuple(errors),
        exact_h=exact_h,
        exact_u=exact_u,
        exact_robust_residual=exact_residual,
        exact_absolute_gap=exact_absolute_gap,
        exact_relative_width=exact_relative_width,
        rounding_provenance_verified=(
            parsed.rounding_provenance is not None
            and not provenance_errors
            and not errors
        ),
        exact_rounding_loss=exact_rounding_loss,
    )


__all__ = [
    "CERTIFICATE_KIND",
    "CERTIFICATE_SCHEMA_VERSION",
    "ROUNDING_PROVENANCE_SCHEMA_VERSION",
    "ExactLPGroupPosition",
    "FixedThetaDualCertificate",
    "GlobalCertificateVerification",
    "GlobalDualCertificate",
    "LPToRoundingProvenance",
    "build_global_certificate",
    "instance_sha256",
    "original_theta_candidates",
    "verify_global_certificate",
]
