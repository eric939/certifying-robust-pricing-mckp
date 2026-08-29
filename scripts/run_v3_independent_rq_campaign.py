#!/usr/bin/env python3
"""Run the frozen Paper-A v3 independent-reference/cost evidence campaign.

The campaign is deliberately bounded: it exercises exact global certificates
on small and medium instances, numerical scaling on larger instances, and the
coverage-to-feasibility calibration claim in repeated one-step experiments.
All pseudo-random streams, grids, cells, exclusions, and reported metrics are
fixed below.  The release contains original binary64 instance snapshots so its
exact certificates do not depend on regenerating transcendental inputs.

This script never overwrites an existing output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robust_mckp import (  # noqa: E402
    Option,
    PricingInstance,
    build_global_certificate,
    from_pricing_data,
    solve,
    verify_global_certificate,
)
from robust_mckp.certificate import certificate_is_feasible, compute_certificate  # noqa: E402
from robust_mckp.greedy import greedy_lp  # noqa: E402
from robust_mckp.hull import build_upper_hull  # noqa: E402
from robust_mckp.utils import EPS  # noqa: E402
from scripts.run_publishable_experiments import solve_full_robust_highs  # noqa: E402


CAMPAIGN_ID = "2026-08-29-paper-a-v3-independent-rq"
DEFAULT_OUTPUT = ROOT / "results" / "release" / CAMPAIGN_ID
KAPPA = 1.25
ALPHA = 0.10

CERTIFICATE_DESIGN: Tuple[Tuple[str, int, int, int, str], ...] = tuple(
    ("favorable", seed, n, 8, gamma_mode)
    for seed in (5101, 5102)
    for n in (20, 40)
    for gamma_mode in ("sqrt", "ten_percent")
) + tuple(
    ("adversarial", seed, n, 3, "sqrt")
    for seed in (6101, 6102)
    for n in (20, 40)
)

SCALING_DESIGN: Tuple[Tuple[str, int, int, int, str], ...] = tuple(
    ("favorable", seed, n, 8, "sqrt")
    for seed in (5201, 5202, 5203)
    for n in (20, 40, 80, 160)
) + tuple(
    ("adversarial", 6201, n, 3, "sqrt") for n in (20, 40, 80, 160)
)

CALIBRATION_CELLS: Tuple[Tuple[str, str], ...] = (
    ("exchangeable_correlated", "valid"),
    ("exchangeable_heavy_tail", "valid"),
    ("ar1", "negative_control"),
    ("change_point", "negative_control"),
    ("menu_dependent", "negative_control"),
)


@dataclass(frozen=True)
class RawPricingData:
    """Economic inputs retained separately from encoded option coefficients."""

    reference_prices: np.ndarray
    unit_costs: np.ndarray
    weights: np.ndarray
    price_menus: Tuple[np.ndarray, ...]
    demands: Tuple[np.ndarray, ...]
    tolerances: np.ndarray
    kappa: float
    regime: str
    seed: int

    @property
    def n(self) -> int:
        return int(self.reference_prices.size)

    @property
    def m(self) -> int:
        return int(max(len(menu) for menu in self.price_menus))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(relative: str) -> str:
    return _sha256(ROOT / relative)


def _gamma(n: int, mode: str) -> int:
    if mode == "zero":
        return 0
    if mode == "sqrt":
        return max(1, int(math.floor(math.sqrt(n))))
    if mode == "ten_percent":
        return max(1, int(math.floor(0.10 * n)))
    if mode == "full":
        return int(n)
    raise ValueError("unknown Gamma mode: %s" % mode)


def _bounded_price_grid(m: int, sigma: float, phase: float) -> np.ndarray:
    if m < 2:
        raise ValueError("m must be at least two")
    base = np.linspace(-sigma, sigma, int(m), dtype=float)
    if m > 3:
        # Deterministic interior perturbation avoids artificial equal spacing,
        # while endpoints remain exactly on the admissibility boundary.
        perturb = 0.035 * sigma * np.sin(np.arange(m, dtype=float) + phase)
        perturb[0] = perturb[-1] = 0.0
        base = np.sort(np.clip(base + perturb, -sigma, sigma))
    return base


def generate_pricing_data(
    *, seed: int, n: int, m: int, regime: str, kappa: float = KAPPA
) -> RawPricingData:
    """Generate one pricing-structured instance with separate r and q inputs.

    Both regimes contain menu options on either side of ``p = kappa*q``.
    ``adversarial`` additionally creates a dominant exposure and coarse menu,
    deliberately violating the bounded-jump intuition behind relative decay.
    """

    if regime not in {"favorable", "adversarial"}:
        raise ValueError("regime must be favorable or adversarial")
    rng = np.random.default_rng(int(seed))
    sigma = 0.15
    reference = rng.lognormal(mean=math.log(100.0), sigma=0.25, size=n)
    if regime == "favorable":
        cost_ratio = rng.uniform(0.76, 0.84, size=n)
        weights = rng.lognormal(mean=0.0, sigma=0.25, size=n)
        baseline = rng.lognormal(mean=math.log(12.0), sigma=0.35, size=n)
        elasticities = rng.uniform(1.2, 2.8, size=n)
    else:
        cost_ratio = rng.uniform(0.77, 0.83, size=n)
        weights = rng.lognormal(mean=0.0, sigma=0.60, size=n)
        baseline = rng.lognormal(mean=math.log(10.0), sigma=0.70, size=n)
        elasticities = rng.uniform(1.1, 3.2, size=n)
        # One group can contribute a nonvanishing share of total revenue.  This
        # is an intentional boundary case for relative certificate decay.
        weights[0] *= float(n)
        baseline[0] *= max(2.0, float(n) / 2.0)
    unit_costs = reference * cost_ratio

    menus: List[np.ndarray] = []
    demands: List[np.ndarray] = []
    for i in range(n):
        offsets = _bounded_price_grid(m, sigma, phase=0.13 * (i + 1))
        prices = reference[i] * (1.0 + offsets)
        demand = baseline[i] * np.power(prices / reference[i], -elasticities[i])
        menus.append(prices.astype(float))
        demands.append(demand.astype(float))

    return RawPricingData(
        reference_prices=reference.astype(float),
        unit_costs=unit_costs.astype(float),
        weights=weights.astype(float),
        price_menus=tuple(menus),
        demands=tuple(demands),
        tolerances=np.full(n, sigma, dtype=float),
        kappa=float(kappa),
        regime=regime,
        seed=int(seed),
    )


def build_instance(raw: RawPricingData, *, gamma: int, radius: float) -> PricingInstance:
    """Encode sign-aware adverse widths without changing the pricing model."""

    if not (0.0 <= radius < 1.0):
        raise ValueError("radius must lie in [0,1)")
    adverse_widths: List[np.ndarray] = []
    for i, (prices, demand) in enumerate(zip(raw.price_menus, raw.demands)):
        unit_surplus = prices - raw.kappa * raw.unit_costs[i]
        widths = np.where(
            unit_surplus >= 0.0,
            radius * demand,
            (radius / (1.0 - radius)) * demand if radius > 0.0 else 0.0,
        )
        adverse_widths.append(np.asarray(widths, dtype=float))
    return from_pricing_data(
        reference_prices=raw.reference_prices,
        unit_costs=raw.unit_costs,
        weights=raw.weights,
        price_menus=raw.price_menus,
        demands=raw.demands,
        uncertainties=adverse_widths,
        margin_target=raw.kappa,
        tolerances=raw.tolerances,
        gamma=int(gamma),
    )


def _direct_residual(instance: PricingInstance, selections: Sequence[int]) -> float:
    return float(compute_certificate(instance, selections))


def _round_down_selection(lp: Any, hulls: Sequence[Any]) -> List[int]:
    selections: List[int] = []
    for pos, hull in zip(lp.positions, hulls):
        vertex = pos.upper_vertex if pos.lambda_ >= 1.0 - EPS else pos.lower_vertex
        selections.append(int(hull.option_indices[int(vertex)]))
    return selections


def floating_full_family_interval(instance: PricingInstance) -> Dict[str, Any]:
    """Numerically reconstruct the full LP family and literal round-down track.

    This routine is used only for diagnostics and scaling.  The released exact
    certificates are built and verified separately in rational arithmetic.
    """

    candidates = sorted(
        {0.0}.union(
            abs(float(option.uncertainty))
            for group in instance.items
            for option in group
        )
    )
    best_u = -math.inf
    best_u_theta = float("nan")
    best_u_round = -math.inf
    best_u_rounding_loss = float("nan")
    best_u_fractional_jump = 0.0
    best_literal_h = -math.inf
    best_literal_selection: Optional[List[int]] = None
    feasible_thresholds = 0

    for theta in candidates:
        hulls = []
        capacity = -float(instance.gamma) * float(theta)
        for group in instance.items:
            values = np.array([option.value for option in group], dtype=float)
            margins = np.array([option.margin for option in group], dtype=float)
            deviations = np.abs(
                np.array([option.uncertainty for option in group], dtype=float)
            )
            transformed = margins - np.maximum(0.0, deviations - theta)
            star = float(np.max(transformed))
            capacity += star
            costs = np.maximum(star - transformed, 0.0)
            hulls.append(
                build_upper_hull(costs, values, np.arange(values.size, dtype=int))
            )
        if capacity < -1e-10:
            continue
        feasible_thresholds += 1
        lp = greedy_lp(hulls, max(0.0, float(capacity)))
        selection = _round_down_selection(lp, hulls)
        literal_value = float(
            sum(instance.items[i][j].value for i, j in enumerate(selection))
        )
        residual = _direct_residual(instance, selection)
        if residual < -1e-7:
            raise AssertionError("literal round-down failed direct robust feasibility")
        if literal_value > best_literal_h:
            best_literal_h = literal_value
            best_literal_selection = selection

        lp_value = float(lp.lp_value)
        if lp_value > best_u:
            fractional_jump = 0.0
            fractional_count = 0
            for pos, hull in zip(lp.positions, hulls):
                if EPS < pos.lambda_ < 1.0 - EPS:
                    fractional_count += 1
                    fractional_jump = float(
                        hull.values[int(pos.upper_vertex)]
                        - hull.values[int(pos.lower_vertex)]
                    )
            if fractional_count > 1:
                raise AssertionError("continuous MCKP solution has multiple fractional groups")
            best_u = lp_value
            best_u_theta = float(theta)
            best_u_round = literal_value
            best_u_rounding_loss = lp_value - literal_value
            best_u_fractional_jump = max(0.0, fractional_jump)

    if best_literal_selection is None or not math.isfinite(best_u):
        raise RuntimeError("no feasible threshold in full-family sweep")
    if best_u_rounding_loss > best_u_fractional_jump + 1e-6:
        raise AssertionError("rounding loss exceeds its adjacent fractional jump")
    return {
        "U_float": best_u,
        "theta_U_float": best_u_theta,
        "round_at_theta_U": best_u_round,
        "rounding_loss_at_theta_U": best_u_rounding_loss,
        "fractional_jump_at_theta_U": best_u_fractional_jump,
        "literal_H_float": best_literal_h,
        "literal_selection": best_literal_selection,
        "threshold_count": len(candidates),
        "feasible_threshold_count": feasible_thresholds,
    }


def _hex_or_none(value: Optional[float]) -> Optional[str]:
    return None if value is None else float(value).hex()


def instance_payload(
    instance: PricingInstance,
    raw: RawPricingData,
    *,
    gamma_mode: str,
    radius: float,
) -> Dict[str, Any]:
    """Serialize economic inputs and authoritative binary64 option data."""

    return {
        "schema_version": 1,
        "coefficient_interpretation": "exact binary64 hexadecimal",
        "regime": raw.regime,
        "seed": raw.seed,
        "n": raw.n,
        "m": raw.m,
        "gamma": int(instance.gamma),
        "gamma_mode": gamma_mode,
        "kappa": float(raw.kappa).hex(),
        "radius": float(radius).hex(),
        "reference_prices": [float(value).hex() for value in raw.reference_prices],
        "unit_costs": [float(value).hex() for value in raw.unit_costs],
        "weights": [float(value).hex() for value in raw.weights],
        "tolerances": [float(value).hex() for value in raw.tolerances],
        "groups": [
            [
                {
                    "value": float(option.value).hex(),
                    "margin": float(option.margin).hex(),
                    "uncertainty": float(option.uncertainty).hex(),
                    "price": _hex_or_none(option.price),
                }
                for option in group
            ]
            for group in instance.items
        ],
    }


def load_instance_payload(payload: Mapping[str, Any]) -> PricingInstance:
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported instance schema")
    groups = []
    for raw_group in payload["groups"]:
        group = []
        for raw_option in raw_group:
            price_text = raw_option["price"]
            group.append(
                Option(
                    value=float.fromhex(raw_option["value"]),
                    margin=float.fromhex(raw_option["margin"]),
                    uncertainty=float.fromhex(raw_option["uncertainty"]),
                    price=(None if price_text is None else float.fromhex(price_text)),
                )
            )
        groups.append(group)
    return PricingInstance(
        items=groups,
        gamma=int(payload["gamma"]),
        name="%s-seed-%s" % (payload["regime"], payload["seed"]),
    )


def _mixed_sign_counts(instance: PricingInstance) -> Tuple[int, int]:
    negative = sum(option.margin < 0.0 for group in instance.items for option in group)
    positive = sum(option.margin > 0.0 for group in instance.items for option in group)
    return int(negative), int(positive)


def _safe_relative_width(u: float, h: float) -> float:
    return (u - h) / u if u > 0.0 else float("nan")


def run_certificate_study(output: Path, *, quick: bool) -> List[Dict[str, Any]]:
    design = CERTIFICATE_DESIGN[:3] if quick else CERTIFICATE_DESIGN
    rows: List[Dict[str, Any]] = []
    for regime, seed, n, m, gamma_mode in design:
        gamma = _gamma(n, gamma_mode)
        radius = 0.12 if regime == "favorable" else 0.22
        raw = generate_pricing_data(seed=seed, n=n, m=m, regime=regime)
        instance = build_instance(raw, gamma=gamma, radius=radius)
        solution = solve(instance, upgrade_completion=True)
        if not solution.is_feasible or not certificate_is_feasible(
            instance, solution.selections
        ):
            raise RuntimeError("campaign solver did not return a robust-feasible plan")
        sweep = floating_full_family_interval(instance)
        certificate = build_global_certificate(
            instance,
            solution.selections,
            include_rounding_provenance=True,
        )
        serialized = certificate.to_json(indent=2) + "\n"
        verification = verify_global_certificate(instance, serialized)
        if not verification.valid:
            raise RuntimeError("exact certificate verification failed")

        key = "%s_seed-%d_n-%d_m-%d_gamma-%d" % (
            regime,
            seed,
            n,
            m,
            gamma,
        )
        instance_path = output / "certificate-study" / "instances" / (key + ".json")
        certificate_path = output / "certificate-study" / "certificates" / (key + ".json")
        instance_path.parent.mkdir(parents=True, exist_ok=True)
        certificate_path.parent.mkdir(parents=True, exist_ok=True)
        instance_path.write_text(
            json.dumps(
                instance_payload(
                    instance, raw, gamma_mode=gamma_mode, radius=radius
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        certificate_path.write_text(serialized, encoding="utf-8")

        highs = solve_full_robust_highs(instance, time_limit=30.0)
        highs_objective = float(highs.get("objective", float("nan")))
        exact_width = (
            float(certificate.relative_width)
            if certificate.relative_width is not None
            else float("nan")
        )
        exact_u = float(certificate.u)
        exact_h = float(certificate.h)
        provenance = certificate.rounding_provenance
        if provenance is None or not verification.rounding_provenance_verified:
            raise RuntimeError("exact LP-to-rounding provenance verification failed")
        negative, positive = _mixed_sign_counts(instance)
        rows.append(
            {
                "key": key,
                "regime": regime,
                "seed": seed,
                "n": n,
                "m": m,
                "gamma_mode": gamma_mode,
                "gamma": gamma,
                "radius": radius,
                "negative_margin_options": negative,
                "positive_margin_options": positive,
                "threshold_count": len(certificate.thresholds),
                "feasible_threshold_count": sum(
                    record.feasible for record in certificate.thresholds
                ),
                "literal_H_float": float(sweep["literal_H_float"]),
                "completed_H_exact": exact_h,
                "U_exact": exact_u,
                "literal_relative_width_float": _safe_relative_width(
                    exact_u, float(sweep["literal_H_float"])
                ),
                "completed_relative_width_exact": exact_width,
                "rounding_loss_at_theta_U_float": float(
                    sweep["rounding_loss_at_theta_U"]
                ),
                "fractional_jump_at_theta_U_float": float(
                    sweep["fractional_jump_at_theta_U"]
                ),
                "robust_residual_exact": float(certificate.robust_residual),
                "certificate_bytes": len(serialized.encode("utf-8")),
                "certificate_verified": bool(verification.valid),
                "certificate_schema_version": int(certificate.schema_version),
                "rounding_provenance_verified": bool(
                    verification.rounding_provenance_verified
                ),
                "exact_rounding_loss": float(provenance.rounding_loss),
                "exact_adjacent_value_jump": float(
                    provenance.adjacent_value_jump
                ),
                "exact_gap_leq_rounding_loss": bool(
                    certificate.absolute_gap <= provenance.rounding_loss
                ),
                "highs_status": str(highs.get("status", "UNKNOWN")),
                "highs_certified": bool(highs.get("certified", False)),
                "highs_objective": highs_objective,
                "completed_highs_absolute_difference": (
                    abs(exact_h - highs_objective)
                    if math.isfinite(highs_objective)
                    else float("nan")
                ),
                "instance_file": str(instance_path.relative_to(output)),
                "instance_sha256": _sha256(instance_path),
                "certificate_file": str(certificate_path.relative_to(output)),
                "certificate_sha256": _sha256(certificate_path),
            }
        )
    return rows


def run_scaling_study(*, quick: bool) -> List[Dict[str, Any]]:
    design = SCALING_DESIGN[:4] if quick else SCALING_DESIGN
    rows: List[Dict[str, Any]] = []
    for regime, seed, n, m, gamma_mode in design:
        gamma = _gamma(n, gamma_mode)
        radius = 0.12 if regime == "favorable" else 0.22
        raw = generate_pricing_data(seed=seed, n=n, m=m, regime=regime)
        instance = build_instance(raw, gamma=gamma, radius=radius)
        solution = solve(instance, upgrade_completion=True)
        if not solution.is_feasible:
            raise RuntimeError("scaling solver returned no feasible plan")
        sweep = floating_full_family_interval(instance)
        u = float(sweep["U_float"])
        h = float(solution.objective)
        negative, positive = _mixed_sign_counts(instance)
        rows.append(
            {
                "regime": regime,
                "seed": seed,
                "n": n,
                "m": m,
                "gamma_mode": gamma_mode,
                "gamma": gamma,
                "radius": radius,
                "negative_margin_options": negative,
                "positive_margin_options": positive,
                "threshold_count": int(sweep["threshold_count"]),
                "feasible_threshold_count": int(sweep["feasible_threshold_count"]),
                "literal_H_float": float(sweep["literal_H_float"]),
                "completed_H_float": h,
                "U_float": u,
                "literal_relative_width_float": _safe_relative_width(
                    u, float(sweep["literal_H_float"])
                ),
                "completed_relative_width_float": _safe_relative_width(u, h),
                "n_times_completed_relative_width": n * _safe_relative_width(u, h),
                "rounding_loss_at_theta_U_float": float(
                    sweep["rounding_loss_at_theta_U"]
                ),
                "fractional_jump_at_theta_U_float": float(
                    sweep["fractional_jump_at_theta_U"]
                ),
                "objective_per_item": h / n,
                "robust_residual": _direct_residual(instance, solution.selections),
                "certificate_feasible": bool(
                    certificate_is_feasible(instance, solution.selections)
                ),
                "interpretation": "floating diagnostic; not an exact certificate",
            }
        )
    return rows


def run_gamma_frontier(output: Path, *, quick: bool) -> List[Dict[str, Any]]:
    """Certify every integer Gamma for one fixed favorable economic instance."""

    seed = 5301
    n = 20
    m = 8
    radius = 0.12
    gamma_values = (0, 5, 10, 15, 20) if quick else tuple(range(n + 1))
    raw = generate_pricing_data(seed=seed, n=n, m=m, regime="favorable")
    rows: List[Dict[str, Any]] = []
    for gamma in gamma_values:
        instance = build_instance(raw, gamma=gamma, radius=radius)
        solution = solve(instance, upgrade_completion=True)
        if not solution.is_feasible:
            raise RuntimeError("Gamma-frontier instance is infeasible")
        certificate = build_global_certificate(
            instance,
            solution.selections,
            include_rounding_provenance=True,
        )
        serialized = certificate.to_json(indent=2) + "\n"
        verification = verify_global_certificate(instance, serialized)
        if not verification.valid or not verification.rounding_provenance_verified:
            raise RuntimeError("Gamma-frontier certificate verification failed")
        provenance = certificate.rounding_provenance
        assert provenance is not None
        key = "favorable_seed-%d_n-%d_m-%d_gamma-%d" % (
            seed,
            n,
            m,
            gamma,
        )
        instance_path = output / "gamma-frontier" / "instances" / (key + ".json")
        certificate_path = output / "gamma-frontier" / "certificates" / (key + ".json")
        instance_path.parent.mkdir(parents=True, exist_ok=True)
        certificate_path.parent.mkdir(parents=True, exist_ok=True)
        instance_path.write_text(
            json.dumps(
                instance_payload(
                    instance,
                    raw,
                    gamma_mode="frontier_integer",
                    radius=radius,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        certificate_path.write_text(serialized, encoding="utf-8")
        rows.append(
            {
                "key": key,
                "seed": seed,
                "n": n,
                "m": m,
                "gamma": gamma,
                "radius": radius,
                "H_exact": float(certificate.h),
                "U_exact": float(certificate.u),
                "absolute_gap_exact": float(certificate.absolute_gap),
                "relative_width_exact": (
                    float(certificate.relative_width)
                    if certificate.relative_width is not None
                    else float("nan")
                ),
                "robust_residual_exact": float(certificate.robust_residual),
                "exact_rounding_loss": float(provenance.rounding_loss),
                "exact_gap_leq_rounding_loss": bool(
                    certificate.absolute_gap <= provenance.rounding_loss
                ),
                "certificate_verified": bool(verification.valid),
                "rounding_provenance_verified": bool(
                    verification.rounding_provenance_verified
                ),
                "instance_file": str(instance_path.relative_to(output)),
                "instance_sha256": _sha256(instance_path),
                "certificate_file": str(certificate_path.relative_to(output)),
                "certificate_sha256": _sha256(certificate_path),
            }
        )
    nominal_h = float(rows[0]["H_exact"])
    for row in rows:
        row["revenue_ratio_vs_gamma_zero"] = (
            float(row["H_exact"]) / nominal_h if nominal_h > 0.0 else float("nan")
        )
        row["revenue_cost_vs_gamma_zero"] = (
            1.0 - float(row["revenue_ratio_vs_gamma_zero"])
        )
    return rows


def _order_quantile(values: Sequence[float], probability: float) -> float:
    """Deterministic upper empirical quantile without interpolation."""

    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    ordered = sorted(float(value) for value in values)
    rank = max(1, int(math.ceil(probability * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]


def _conformal_radius(scores: Sequence[float], alpha: float) -> float:
    """Split-conformal order statistic with finite-sample correction."""

    ordered = sorted(float(score) for score in scores)
    if not ordered:
        raise ValueError("conformal calibration requires at least one score")
    rank = int(math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return float("inf") if rank > len(ordered) else ordered[rank - 1]


def _uncertainty_score(residual: np.ndarray, gamma: int) -> float:
    if gamma <= 0:
        raise ValueError("Gamma must be positive for calibration scoring")
    absolute = np.abs(np.asarray(residual, dtype=float))
    return float(max(np.max(absolute), np.sum(absolute) / float(gamma)))


def _draw_residual_path(
    *, cell: str, rng: np.random.Generator, n: int, length: int
) -> np.ndarray:
    """Draw bounded residuals for a declared positive or negative control."""

    if length < 1:
        raise ValueError("length must be positive")
    if cell in {"exchangeable_correlated", "change_point", "menu_dependent"}:
        common = rng.normal(size=(length, 1))
        idiosyncratic = rng.normal(size=(length, n))
        latent = 0.62 * common + math.sqrt(1.0 - 0.62**2) * idiosyncratic
        path = np.tanh(0.105 * latent)
    elif cell == "exchangeable_heavy_tail":
        common = rng.standard_t(df=3.0, size=(length, 1)) / math.sqrt(3.0)
        idiosyncratic = rng.standard_t(df=3.0, size=(length, n)) / math.sqrt(3.0)
        latent = 0.48 * common + math.sqrt(1.0 - 0.48**2) * idiosyncratic
        path = np.tanh(0.115 * latent)
    elif cell == "ar1":
        phi = 0.90
        common = np.empty(length, dtype=float)
        idiosyncratic = np.empty((length, n), dtype=float)
        common[0] = rng.normal()
        idiosyncratic[0] = rng.normal(size=n)
        innovation_scale = math.sqrt(1.0 - phi**2)
        for t in range(1, length):
            common[t] = phi * common[t - 1] + innovation_scale * rng.normal()
            idiosyncratic[t] = (
                phi * idiosyncratic[t - 1]
                + innovation_scale * rng.normal(size=n)
            )
        latent = 0.62 * common[:, None] + math.sqrt(1.0 - 0.62**2) * idiosyncratic
        path = np.tanh(0.105 * latent)
    else:
        raise ValueError("unknown calibration cell: %s" % cell)
    return np.asarray(path, dtype=float)


def _tune_gamma(tune_residuals: np.ndarray) -> int:
    """Tune an effective shock-support budget on a disjoint data block."""

    absolute = np.abs(np.asarray(tune_residuals, dtype=float))
    maxima = np.max(absolute, axis=1)
    effective_support = np.sum(absolute, axis=1) / np.maximum(maxima, 1e-12)
    tuned = int(math.ceil(_order_quantile(effective_support.tolist(), 0.90)))
    return max(1, min(int(absolute.shape[1]), tuned))


def _policy_test_residual(
    *,
    cell: str,
    raw: RawPricingData,
    selections: Sequence[int],
    base_residual: np.ndarray,
) -> np.ndarray:
    residual = np.asarray(base_residual, dtype=float).copy()
    if cell == "change_point":
        # The shift is fixed by item-level economics, not by the decision.
        directions = []
        for i in range(raw.n):
            middle = len(raw.price_menus[i]) // 2
            surplus = raw.price_menus[i][middle] - raw.kappa * raw.unit_costs[i]
            directions.append(-1.0 if surplus >= 0.0 else 1.0)
        residual += 0.50 * np.asarray(directions, dtype=float)
    elif cell == "menu_dependent":
        # This deliberately violates policy-independent shock generation.
        for i, option in enumerate(selections):
            surplus = (
                raw.price_menus[i][int(option)] - raw.kappa * raw.unit_costs[i]
            )
            residual[i] += -0.25 if surplus >= 0.0 else 0.25
    return np.clip(residual, -0.90, 0.90)


def _realized_policy_metrics(
    raw: RawPricingData,
    selections: Sequence[int],
    residual: np.ndarray,
) -> Dict[str, float]:
    realized_margin = 0.0
    realized_revenue = 0.0
    for i, option in enumerate(selections):
        j = int(option)
        shock = float(residual[i])
        nominal_demand = float(raw.demands[i][j])
        realized_demand = (
            nominal_demand * (1.0 + shock)
            if shock <= 0.0
            else nominal_demand / (1.0 - shock)
        )
        price = float(raw.price_menus[i][j])
        weight = float(raw.weights[i])
        realized_revenue += weight * price * realized_demand
        realized_margin += (
            weight
            * (price - raw.kappa * float(raw.unit_costs[i]))
            * realized_demand
        )
    return {
        "realized_margin": float(realized_margin),
        "realized_revenue": float(realized_revenue),
    }


def _selected_negative_count(raw: RawPricingData, selections: Sequence[int]) -> int:
    return int(
        sum(
            raw.price_menus[i][int(option)] - raw.kappa * raw.unit_costs[i] < 0.0
            for i, option in enumerate(selections)
        )
    )


def run_calibration_study(*, quick: bool) -> List[Dict[str, Any]]:
    """Run split-calibration controls and assumption stress tests."""

    repetitions = 5 if quick else 120
    tune_size = 40
    calibration_size = 60
    n = 12
    m = 5
    rows: List[Dict[str, Any]] = []
    for cell_index, (cell, role) in enumerate(CALIBRATION_CELLS):
        for repetition in range(repetitions):
            seed = 710000 + 10000 * cell_index + repetition
            rng = np.random.default_rng(np.random.SeedSequence([seed, 29082026]))
            raw = generate_pricing_data(seed=seed, n=n, m=m, regime="favorable")
            path = _draw_residual_path(
                cell=cell,
                rng=rng,
                n=n,
                length=tune_size + calibration_size + 1,
            )
            tune = path[:tune_size]
            calibration = path[tune_size : tune_size + calibration_size]
            base_test = path[-1]
            gamma = _tune_gamma(tune)
            calibration_scores = [
                _uncertainty_score(row, gamma) for row in calibration
            ]
            radius = _conformal_radius(calibration_scores, ALPHA)
            if not math.isfinite(radius) or radius >= 1.0:
                raise RuntimeError("calibrated radius is not finite and bounded")

            instances = {
                "nominal": build_instance(raw, gamma=0, radius=radius),
                "robust": build_instance(raw, gamma=gamma, radius=radius),
                "box": build_instance(raw, gamma=n, radius=radius),
            }
            solutions = {
                policy: solve(instance, upgrade_completion=True)
                for policy, instance in instances.items()
            }
            row: Dict[str, Any] = {
                "cell": cell,
                "role": role,
                "repetition": repetition,
                "seed": seed,
                "n": n,
                "m": m,
                "tune_size": tune_size,
                "calibration_size": calibration_size,
                "alpha": ALPHA,
                "gamma": gamma,
                "radius": radius,
            }
            robust_test: Optional[np.ndarray] = None
            for policy, solution in solutions.items():
                feasible = bool(solution.is_feasible)
                row[policy + "_solution_feasible"] = feasible
                if not feasible:
                    row[policy + "_objective"] = float("nan")
                    row[policy + "_realized_margin"] = float("nan")
                    row[policy + "_realized_revenue"] = float("nan")
                    row[policy + "_violation"] = False
                    row[policy + "_negative_selected"] = -1
                    continue
                policy_test = _policy_test_residual(
                    cell=cell,
                    raw=raw,
                    selections=solution.selections,
                    base_residual=base_test,
                )
                if policy == "robust":
                    robust_test = policy_test
                realized = _realized_policy_metrics(raw, solution.selections, policy_test)
                tolerance = 1e-8 * max(1.0, abs(float(solution.objective)))
                row[policy + "_objective"] = float(solution.objective)
                row[policy + "_realized_margin"] = realized["realized_margin"]
                row[policy + "_realized_revenue"] = realized["realized_revenue"]
                row[policy + "_violation"] = bool(
                    realized["realized_margin"] < -tolerance
                )
                row[policy + "_negative_selected"] = _selected_negative_count(
                    raw, solution.selections
                )
                row[policy + "_model_certificate_residual"] = _direct_residual(
                    instances[policy], solution.selections
                )

            if robust_test is None:
                row["test_score"] = float("nan")
                row["set_covered"] = False
            else:
                score = _uncertainty_score(robust_test, gamma)
                covered = bool(score <= radius + 1e-12)
                row["test_score"] = score
                row["set_covered"] = covered
                if covered and bool(row["robust_violation"]):
                    raise AssertionError(
                        "covered realization violates the robust constraint"
                    )
            nominal_objective = float(row["nominal_objective"])
            for policy in ("robust", "box"):
                objective = float(row[policy + "_objective"])
                row[policy + "_revenue_ratio_vs_nominal"] = (
                    objective / nominal_objective
                    if nominal_objective > 0.0 and math.isfinite(objective)
                    else float("nan")
                )
            rows.append(row)
    return rows


def _finite(values: Sequence[Any]) -> List[float]:
    result = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _median(values: Sequence[Any]) -> float:
    finite = _finite(values)
    return float(statistics.median(finite)) if finite else float("nan")


def _mean(values: Sequence[Any]) -> float:
    finite = _finite(values)
    return float(statistics.fmean(finite)) if finite else float("nan")


def _rate(values: Sequence[Any]) -> float:
    values_list = list(values)
    def as_bool(value: Any) -> bool:
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return bool(value)
    return (
        float(sum(as_bool(value) for value in values_list) / len(values_list))
        if values_list
        else float("nan")
    )


def _wilson(
    successes: int, total: int, z: float = 1.959963984540054
) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total**2)
        )
        / denominator
    )
    return float(center - half), float(center + half)


def summarize_campaign(
    certificate_rows: Sequence[Mapping[str, Any]],
    scaling_rows: Sequence[Mapping[str, Any]],
    frontier_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    certificate_summary: Dict[str, Any] = {
        "instances": len(certificate_rows),
        "all_exact_certificates_verified": bool(certificate_rows)
        and all(bool(row["certificate_verified"]) for row in certificate_rows),
        "all_rounding_provenance_verified": bool(certificate_rows)
        and all(bool(row["rounding_provenance_verified"]) for row in certificate_rows),
        "all_gap_leq_literal_rounding_loss": bool(certificate_rows)
        and all(bool(row["exact_gap_leq_rounding_loss"]) for row in certificate_rows),
        "all_instances_independent_r_q": True,
        "all_instances_mixed_sign": bool(certificate_rows)
        and all(
            int(row["negative_margin_options"]) > 0
            and int(row["positive_margin_options"]) > 0
            for row in certificate_rows
        ),
        "by_regime": {},
    }
    for regime in ("favorable", "adversarial"):
        subset = [row for row in certificate_rows if row["regime"] == regime]
        if not subset:
            continue
        widths = _finite(
            [row["completed_relative_width_exact"] for row in subset]
        )
        highs_differences = _finite(
            [row["completed_highs_absolute_difference"] for row in subset]
        )
        certificate_summary["by_regime"][regime] = {
            "instances": len(subset),
            "median_completed_relative_width": _median(widths),
            "maximum_completed_relative_width": max(widths),
            "median_literal_relative_width": _median(
                [row["literal_relative_width_float"] for row in subset]
            ),
            "maximum_highs_objective_difference": (
                max(highs_differences) if highs_differences else float("nan")
            ),
            "median_exact_rounding_loss": _median(
                [row["exact_rounding_loss"] for row in subset]
            ),
        }

    scaling_summary: Dict[str, Any] = {"rows": len(scaling_rows), "by_regime_n": {}}
    for regime in ("favorable", "adversarial"):
        regime_points: Dict[str, Any] = {}
        n_values = sorted(
            {int(row["n"]) for row in scaling_rows if row["regime"] == regime}
        )
        for n in n_values:
            subset = [
                row
                for row in scaling_rows
                if row["regime"] == regime and int(row["n"]) == n
            ]
            regime_points[str(n)] = {
                "replicates": len(subset),
                "median_completed_relative_width": _median(
                    [row["completed_relative_width_float"] for row in subset]
                ),
                "median_n_times_width": _median(
                    [row["n_times_completed_relative_width"] for row in subset]
                ),
                "median_objective_per_item": _median(
                    [row["objective_per_item"] for row in subset]
                ),
            }
        if regime_points:
            scaling_summary["by_regime_n"][regime] = regime_points
    favorable_points = scaling_summary["by_regime_n"].get("favorable", {})
    positive_points = [
        (float(n), float(record["median_completed_relative_width"]))
        for n, record in favorable_points.items()
        if float(record["median_completed_relative_width"]) > 0.0
    ]
    scaling_summary["favorable_log_log_slope"] = (
        float(
            np.polyfit(
                np.log([point[0] for point in positive_points]),
                np.log([point[1] for point in positive_points]),
                1,
            )[0]
        )
        if len(positive_points) >= 2
        else float("nan")
    )

    calibration_summary: Dict[str, Any] = {
        "rows": len(calibration_rows),
        "target_miscoverage": ALPHA,
        "by_cell": {},
    }
    for cell, role in CALIBRATION_CELLS:
        subset = [row for row in calibration_rows if row["cell"] == cell]
        if not subset:
            continue
        coverage_successes = sum(bool(row["set_covered"]) for row in subset)
        low, high = _wilson(coverage_successes, len(subset))
        calibration_summary["by_cell"][cell] = {
            "role": role,
            "repetitions": len(subset),
            "median_gamma": _median([row["gamma"] for row in subset]),
            "median_radius": _median([row["radius"] for row in subset]),
            "set_coverage": coverage_successes / len(subset),
            "set_coverage_wilson_95_low": low,
            "set_coverage_wilson_95_high": high,
            "robust_violation_rate": _rate(
                [row["robust_violation"] for row in subset]
            ),
            "nominal_violation_rate": _rate(
                [row["nominal_violation"] for row in subset]
            ),
            "box_violation_rate": _rate([row["box_violation"] for row in subset]),
            "robust_feasibility_rate": _rate(
                [row["robust_solution_feasible"] for row in subset]
            ),
            "mean_robust_revenue_ratio_vs_nominal": _mean(
                [row["robust_revenue_ratio_vs_nominal"] for row in subset]
            ),
            "mean_box_revenue_ratio_vs_nominal": _mean(
                [row["box_revenue_ratio_vs_nominal"] for row in subset]
            ),
            "covered_and_robust_violation_count": sum(
                bool(row["set_covered"]) and bool(row["robust_violation"])
                for row in subset
            ),
        }
    return {
        "campaign_id": CAMPAIGN_ID,
        "certificate_study": certificate_summary,
        "scaling_study": scaling_summary,
        "gamma_frontier": {
            "rows": len(frontier_rows),
            "all_integer_gamma": [int(row["gamma"]) for row in frontier_rows],
            "all_certificates_verified": bool(frontier_rows)
            and all(bool(row["certificate_verified"]) for row in frontier_rows),
            "all_rounding_provenance_verified": bool(frontier_rows)
            and all(
                bool(row["rounding_provenance_verified"])
                for row in frontier_rows
            ),
            "maximum_revenue_cost_vs_gamma_zero": max(
                _finite(
                    [row["revenue_cost_vs_gamma_zero"] for row in frontier_rows]
                )
            ),
        },
        "calibration_study": calibration_summary,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return "" if not math.isfinite(float(value)) else repr(float(value))
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty campaign table")
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _tex_number(value: float, digits: int = 2) -> str:
    return "NA" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def write_macros(path: Path, summary: Mapping[str, Any]) -> None:
    certificate = summary["certificate_study"]
    scaling = summary["scaling_study"]
    frontier = summary["gamma_frontier"]
    calibration = summary["calibration_study"]["by_cell"]
    favorable = certificate["by_regime"].get("favorable", {})
    adversarial = certificate["by_regime"].get("adversarial", {})
    verified_count = (
        certificate["instances"]
        if certificate["all_exact_certificates_verified"]
        else 0
    )
    lines = [
        "% Generated by scripts/run_v3_independent_rq_campaign.py.",
        "\\newcommand{\\PaperAVThreeCertificateInstances}{%d}"
        % certificate["instances"],
        "\\newcommand{\\PaperAVThreeExactVerified}{%d}" % verified_count,
        "\\newcommand{\\PaperAVThreeMixedSignInstances}{%d}"
        % (
            certificate["instances"]
            if certificate["all_instances_mixed_sign"]
            else 0
        ),
        "\\newcommand{\\PaperAVThreeFavorableMedianGapPct}{%s}"
        % _tex_number(
            100.0
            * float(favorable.get("median_completed_relative_width", float("nan"))),
            3,
        ),
        "\\newcommand{\\PaperAVThreeAdversarialMedianGapPct}{%s}"
        % _tex_number(
            100.0
            * float(adversarial.get("median_completed_relative_width", float("nan"))),
            3,
        ),
        "\\newcommand{\\PaperAVThreeFavorableScalingSlope}{%s}"
        % _tex_number(float(scaling["favorable_log_log_slope"]), 2),
        "\\newcommand{\\PaperAVThreeGammaFrontierPoints}{%d}" % frontier["rows"],
        "\\newcommand{\\PaperAVThreeGammaFrontierMaxRevenueCostPct}{%s}"
        % _tex_number(
            100.0 * float(frontier["maximum_revenue_cost_vs_gamma_zero"]), 2
        ),
    ]
    macro_names = {
        "exchangeable_correlated": "ExchangeableCorrelated",
        "exchangeable_heavy_tail": "ExchangeableHeavyTail",
        "ar1": "AROne",
        "change_point": "ChangePoint",
        "menu_dependent": "MenuDependent",
    }
    for cell, macro in macro_names.items():
        record = calibration.get(cell)
        if record is None:
            continue
        lines.extend(
            [
                "\\newcommand{\\PaperAVThree%sCoveragePct}{%s}"
                % (macro, _tex_number(100.0 * float(record["set_coverage"]), 1)),
                "\\newcommand{\\PaperAVThree%sRobustViolationPct}{%s}"
                % (
                    macro,
                    _tex_number(100.0 * float(record["robust_violation_rate"]), 1),
                ),
                "\\newcommand{\\PaperAVThree%sNominalViolationPct}{%s}"
                % (
                    macro,
                    _tex_number(100.0 * float(record["nominal_violation_rate"]), 1),
                ),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def write_certificate_figure(
    path: Path,
    certificate_rows: Sequence[Mapping[str, Any]],
    scaling_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_plot_style()
    colors = {"favorable": "#0072B2", "adversarial": "#D55E00"}
    display_labels = {
        "favorable": "Favorable",
        "adversarial": "Dominant-group",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))
    regimes = [
        regime
        for regime in ("favorable", "adversarial")
        if any(row["regime"] == regime for row in certificate_rows)
    ]
    data = [
        [
            100.0 * float(row["completed_relative_width_exact"])
            for row in certificate_rows
            if row["regime"] == regime
        ]
        for regime in regimes
    ]
    boxes = axes[0].boxplot(
        data,
        tick_labels=[display_labels[regime] for regime in regimes],
        patch_artist=True,
        widths=0.55,
    )
    for box, regime in zip(boxes["boxes"], regimes):
        box.set_facecolor(colors[regime])
        box.set_alpha(0.72)
    axes[0].set_ylabel("Exact certificate width (%)")
    axes[0].set_title("(a) Exact completed intervals")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.6)

    for regime in regimes:
        n_values = sorted(
            {int(row["n"]) for row in scaling_rows if row["regime"] == regime}
        )
        medians = [
            _median(
                [
                    row["completed_relative_width_float"]
                    for row in scaling_rows
                    if row["regime"] == regime and int(row["n"]) == n
                ]
            )
            for n in n_values
        ]
        axes[1].plot(
            n_values,
            [max(value, 1e-8) for value in medians],
            marker="D" if regime == "adversarial" else "o",
            linestyle="none" if regime == "adversarial" else "-",
            linewidth=1.4,
            markersize=4.0,
            color=colors[regime],
            label=display_labels[regime],
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks([20, 40, 80, 160], labels=["20", "40", "80", "160"])
    axes[1].set_xlabel("Number of groups n")
    axes[1].set_ylabel("Numerical relative width")
    axes[1].set_title("(b) Scaling and dominant-group regime")
    axes[1].grid(which="both", color="#dddddd", linewidth=0.6)
    axes[1].legend(frameon=False)
    fig.tight_layout(w_pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        dpi=300,
        metadata={"Software": "Paper-A v3 frozen campaign"},
    )
    plt.close(fig)


def write_calibration_figure(
    path: Path, calibration_rows: Sequence[Mapping[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_plot_style()
    names = [cell for cell, _ in CALIBRATION_CELLS]
    labels = ["Corr. iid", "Heavy tail", "AR(1)", "Change point", "Menu dep."]
    miscoverage = []
    robust_violation = []
    nominal_violation = []
    for cell in names:
        subset = [row for row in calibration_rows if row["cell"] == cell]
        miscoverage.append(1.0 - _rate([row["set_covered"] for row in subset]))
        robust_violation.append(_rate([row["robust_violation"] for row in subset]))
        nominal_violation.append(_rate([row["nominal_violation"] for row in subset]))
    positions = np.arange(len(names), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.15, 2.8))
    ax.bar(
        positions - width,
        100.0 * np.asarray(miscoverage),
        width,
        color="#999999",
        label="Set miscoverage",
    )
    ax.bar(
        positions,
        100.0 * np.asarray(robust_violation),
        width,
        color="#0072B2",
        label="Robust violation",
    )
    ax.bar(
        positions + width,
        100.0 * np.asarray(nominal_violation),
        width,
        color="#D55E00",
        label="Nominal violation",
    )
    ax.axhline(100.0 * ALPHA, color="#222222", linestyle="--", linewidth=0.9)
    ax.axvline(1.5, color="#777777", linestyle=":", linewidth=0.9)
    ax.set_xticks(positions, labels=labels)
    ax.set_ylabel("Empirical rate (%)")
    ax.set_title("Split-calibration coverage and realized constraint violations")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        dpi=300,
        metadata={"Software": "Paper-A v3 frozen campaign"},
    )
    plt.close(fig)


def write_gamma_frontier_figure(
    path: Path, frontier_rows: Sequence[Mapping[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_plot_style()
    gamma = np.asarray([int(row["gamma"]) for row in frontier_rows], dtype=float)
    h = np.asarray([float(row["H_exact"]) for row in frontier_rows], dtype=float)
    u = np.asarray([float(row["U_exact"]) for row in frontier_rows], dtype=float)
    cost = 100.0 * np.asarray(
        [float(row["revenue_cost_vs_gamma_zero"]) for row in frontier_rows],
        dtype=float,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55))
    axes[0].fill_between(gamma, h, u, color="#56B4E9", alpha=0.30, label="[H, U]")
    axes[0].plot(gamma, h, color="#0072B2", marker="o", markersize=3.0, label="H")
    axes[0].plot(gamma, u, color="#D55E00", linewidth=1.2, label="U")
    axes[0].set_xlabel("Budget Gamma")
    axes[0].set_ylabel("Certified revenue")
    axes[0].set_title("(a) Exact certificate path")
    axes[0].grid(color="#dddddd", linewidth=0.6)
    axes[0].legend(frameon=False)
    axes[1].plot(gamma, cost, color="#009E73", marker="o", markersize=3.0)
    axes[1].set_xlabel("Budget Gamma")
    axes[1].set_ylabel("Revenue cost vs. Gamma=0 (%)")
    axes[1].set_title("(b) Incumbent revenue cost")
    axes[1].grid(color="#dddddd", linewidth=0.6)
    fig.tight_layout(w_pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        dpi=300,
        metadata={"Software": "Paper-A v3 frozen campaign"},
    )
    plt.close(fig)


def _git_head() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_readme(path: Path, *, quick: bool, summary: Mapping[str, Any]) -> None:
    certificate = summary["certificate_study"]
    calibration = summary["calibration_study"]
    lines = [
        "# Paper-A v3 independent-reference/cost campaign",
        "",
        "Campaign ID: `%s`." % CAMPAIGN_ID,
        "",
        "This is a deterministic %s campaign frozen before manuscript use. It is not a preregistration."
        % ("quick development" if quick else "full evidence"),
        "",
        "## What this release tests",
        "",
        "- Reference prices `r_i` and unit costs `q_i` are separate and nonidentical; `r_i` controls menu admissibility and `q_i` controls the ratio margin. This is parameter separation, not a claim of stochastic independence.",
        "- Every exact-certificate instance contains both negative- and positive-surplus menu options.",
        "- Favorable instances satisfy bounded-group intuition; dominant-group instances deliberately include one large group.",
        "- Exact schema-v2 rational certificates verify the upper endpoint, incumbent, literal LP round-down witness, and `U-H <=` literal rounding loss. Larger scaling rows are numerical diagnostics, not exact certificates.",
        "- Split calibration is tested under two exchangeable positive controls. AR(1), change-point, and menu-dependent cells are assumption stress tests and carry no coverage guarantee.",
        "",
        "## Released artifacts",
        "",
        "- `certificate-study.csv`: %d exact completed intervals and HiGHS cross-checks."
        % certificate["instances"],
        "- `certificate-study/instances/`: authoritative binary64 snapshots.",
        "- `certificate-study/certificates/`: independently checkable exact certificates.",
        "- `scaling-study.csv`: %d larger numerical full-family sweeps."
        % summary["scaling_study"]["rows"],
        "- `gamma-frontier.csv`: %d exact integer-budget frontier points."
        % summary["gamma_frontier"]["rows"],
        "- `calibration-study.csv`: %d independent tune/calibrate/test repetitions."
        % calibration["rows"],
        "- `summary.json`, `summary-macros.txt`, `figures/`, `protocol.json`, and `manifest.json`.",
        "",
        "## Reproduce and verify",
        "",
        "From a clean checkout with the locked environment installed:",
        "",
        "```bash",
        ".venv/bin/python scripts/run_v3_independent_rq_campaign.py --output-dir results/local/v3-independent-rq",
        ".venv/bin/python scripts/verify_v3_independent_rq_release.py --release-dir results/local/v3-independent-rq",
        "```",
        "",
        "The generator refuses to overwrite an existing directory. Binary64 snapshots, rather than regenerated transcendental inputs, are authoritative for exact verification.",
        "",
        "## Limits",
        "",
        "This is a controlled/semi-synthetic study, not causal evidence of market response or field deployment. The larger scaling table is numerical. Monte Carlo rates have finite sampling uncertainty, shown with Wilson intervals in `summary.json`. Negative controls diagnose assumptions; they do not estimate real-world failure frequencies.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_protocol(path: Path, *, quick: bool) -> None:
    protocol = {
        "campaign_id": CAMPAIGN_ID,
        "frozen_date": "2026-08-29",
        "status": "frozen before manuscript use; not preregistered",
        "quick": bool(quick),
        "source_commit_at_run": _git_head(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "alpha": ALPHA,
        "kappa": KAPPA,
        "independent_r_q_meaning": "separate nonidentical model inputs and code paths; not stochastic independence",
        "certificate_design": [
            list(row) for row in (CERTIFICATE_DESIGN[:3] if quick else CERTIFICATE_DESIGN)
        ],
        "scaling_design": [
            list(row) for row in (SCALING_DESIGN[:4] if quick else SCALING_DESIGN)
        ],
        "gamma_frontier": {
            "seed": 5301,
            "n": 20,
            "m": 8,
            "radius": 0.12,
            "gamma_values": [0, 5, 10, 15, 20]
            if quick
            else list(range(21)),
            "certificate_schema": 2,
        },
        "calibration": {
            "cells": [list(row) for row in CALIBRATION_CELLS],
            "repetitions_per_cell": 5 if quick else 120,
            "n": 12,
            "m": 5,
            "tuning_observations": 40,
            "calibration_observations": 60,
            "test_observations_per_repetition": 1,
            "gamma_rule": "ceil 90th upper empirical quantile of l1/linfinity effective support on tuning block, clipped to [1,n]",
            "radius_rule": "split-conformal ceil((T+1)(1-alpha)) order statistic of max(linfinity,l1/Gamma) on disjoint calibration block",
            "positive_controls": ["exchangeable_correlated", "exchangeable_heavy_tail"],
            "negative_controls": ["ar1", "change_point", "menu_dependent"],
            "negative_control_shifts": {
                "change_point": 0.50,
                "menu_dependent": 0.25,
                "ar1_phi": 0.90,
            },
        },
        "interpretation": {
            "certificate_study": "exact rational verification over authoritative binary64 snapshots",
            "scaling_study": "floating full-family diagnostic; not an exact certificate",
            "calibration_study": "Monte Carlo coverage and realized-feasibility audit",
        },
        "source_sha256": {
            relative: _source_sha256(relative)
            for relative in (
                "scripts/run_v3_independent_rq_campaign.py",
                "scripts/verify_v3_independent_rq_release.py",
                "src/robust_mckp/preprocessing.py",
                "src/robust_mckp/solver.py",
                "src/robust_mckp/global_certificate.py",
            )
            if (ROOT / relative).exists()
        },
    }
    write_json(path, protocol)


def write_manifest(output: Path) -> None:
    records = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            records.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    write_json(
        output / "manifest.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "files": records,
            "file_count": len(records),
        },
    )


def run_campaign(output: Path, *, quick: bool) -> Dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite existing output: %s" % output)
    output.mkdir(parents=True)
    certificate_rows = run_certificate_study(output, quick=quick)
    scaling_rows = run_scaling_study(quick=quick)
    frontier_rows = run_gamma_frontier(output, quick=quick)
    calibration_rows = run_calibration_study(quick=quick)
    write_csv(output / "certificate-study.csv", certificate_rows)
    write_csv(output / "scaling-study.csv", scaling_rows)
    write_csv(output / "gamma-frontier.csv", frontier_rows)
    write_csv(output / "calibration-study.csv", calibration_rows)
    summary = summarize_campaign(
        certificate_rows, scaling_rows, frontier_rows, calibration_rows
    )
    write_json(output / "summary.json", summary)
    write_macros(output / "summary-macros.txt", summary)
    write_certificate_figure(
        output / "figures" / "certificate-and-scaling.png",
        certificate_rows,
        scaling_rows,
    )
    write_calibration_figure(
        output / "figures" / "calibration-controls.png", calibration_rows
    )
    write_gamma_frontier_figure(
        output / "figures" / "gamma-frontier.png", frontier_rows
    )
    write_protocol(output / "protocol.json", quick=quick)
    write_readme(output / "README.md", quick=quick, summary=summary)
    write_manifest(output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--quick", action="store_true", help="run a small development campaign"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_campaign(args.output_dir.resolve(), quick=bool(args.quick))
    print(json.dumps(_json_clean(summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
