#!/usr/bin/env python3
"""Build a transparent Paper-A v3 pilot from Dominick's cereal data.

The script deliberately uses only the Python standard library and NumPy.  It
streams the large movement CSV from its zip archive, chooses one store and a
small portfolio of active cereal UPCs, fits separate predictive own-price
demand curves, calibrates the Gamma set chronologically, solves the robust
finite-menu problem, and evaluates the chosen policy under held-out bounded
relative shocks.  The held-out counterfactual evaluation is semi-synthetic:
shocks observed at historical prices are transported to the selected menu
price and applied to the fixed week-260 nominal-demand vector encoded in the
pricing instance.  It is a scenario stress test, not a week-specific forecast,
causal estimate, or deployment estimate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robust_mckp import Option, PricingInstance, solve  # noqa: E402
from robust_mckp.certificate import compute_certificate  # noqa: E402
from robust_mckp.global_certificate import (  # noqa: E402
    build_global_certificate,
    verify_global_certificate,
)


Key = Tuple[int, int]


@dataclass
class SeriesRow:
    week: int
    demand: float
    price: float
    sale: bool
    profit_pct: float | None


@dataclass
class ItemModel:
    key: Key
    beta: np.ndarray
    cost: float
    reference: float
    menu: np.ndarray
    demand: np.ndarray
    rows: Dict[int, SeriesRow]
    train_rows: int
    tune_rows: int
    log_price_coefficient: float
    baseline_week: int
    menu_training_regular_occurrences: np.ndarray

    def predict(self, price: float, week: int, sale: bool = False) -> float:
        x = _features(price, week, sale)
        return max(1e-6, float(math.expm1(float(x @ self.beta))))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _movement_rows(path: Path) -> Iterable[Mapping[str, str]]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one movement CSV in {path}, found {names}")
        with archive.open(names[0], "r") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                yield from csv.DictReader(text)


def _int(row: Mapping[str, str], name: str, default: int = 0) -> int:
    try:
        return int(float(row.get(name, "")))
    except (TypeError, ValueError):
        return default


def _float(row: Mapping[str, str], name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _features(price: float, week: int, sale: bool) -> np.ndarray:
    angle = 2.0 * math.pi * (week % 52) / 52.0
    return np.array(
        [1.0, math.log(max(price, 1e-8)), float(sale), math.sin(angle), math.cos(angle)],
        dtype=float,
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    idx = max(0, min(len(ordered) - 1, int(math.ceil(probability * len(ordered))) - 1))
    return ordered[idx]


def _calendar_weeks(path: Path) -> Tuple[int, int]:
    minimum: int | None = None
    maximum: int | None = None
    for row in _movement_rows(path):
        week = _int(row, "WEEK", -1)
        if week < 0:
            continue
        minimum = week if minimum is None else min(minimum, week)
        maximum = week if maximum is None else max(maximum, week)
    if minimum is None or maximum is None:
        raise RuntimeError("movement file contains no valid calendar weeks")
    return minimum, maximum


def _first_pass(path: Path, train_end: int) -> Dict[Key, dict]:
    stats: Dict[Key, dict] = {}
    for row in _movement_rows(path):
        if _int(row, "OK") != 1:
            continue
        store = _int(row, "STORE", -1)
        upc = _int(row, "UPC", -1)
        week = _int(row, "WEEK", -1)
        if store < 0 or upc < 0 or week < 0 or week > train_end:
            continue
        key = (store, upc)
        record = stats.setdefault(
            key,
            {
                "valid": 0,
                "positive_price": 0,
                "movement": 0.0,
                "prices": set(),
                "regular_prices": set(),
                "min_week": week,
                "max_week": week,
            },
        )
        record["valid"] += 1
        record["movement"] += max(0.0, _float(row, "MOVE"))
        record["min_week"] = min(record["min_week"], week)
        record["max_week"] = max(record["max_week"], week)
        qty = _float(row, "QTY")
        price = _float(row, "PRICE")
        if qty > 0.0 and price > 0.0:
            unit_price = round(price / qty, 2)
            record["positive_price"] += 1
            if len(record["prices"]) < 128:
                record["prices"].add(unit_price)
            if not str(row.get("SALE", "")).strip() and len(record["regular_prices"]) < 128:
                record["regular_prices"].add(unit_price)
    return stats


def _choose_keys(stats: Mapping[Key, dict], requested: int) -> Tuple[int, List[Key]]:
    by_store: MutableMapping[int, List[Tuple[Key, dict]]] = defaultdict(list)
    for key, record in stats.items():
        span = int(record["max_week"] - record["min_week"] + 1)
        if (
            record["valid"] >= 80
            and record["positive_price"] >= 60
            and len(record["prices"]) >= 4
            and span >= 80
        ):
            by_store[key[0]].append((key, record))
    if not by_store:
        raise RuntimeError("no store has enough active multi-price UPCs under the pilot filters")
    store = max(by_store, key=lambda value: (len(by_store[value]), value))
    ranked = sorted(
        by_store[store],
        key=lambda pair: (
            pair[1]["positive_price"],
            len(pair[1]["regular_prices"]),
            pair[1]["movement"],
            pair[0][1],
        ),
        reverse=True,
    )
    # Fit more candidates than ultimately requested because positive estimated
    # own-price elasticity is an exclusion criterion below.
    return store, [key for key, _ in ranked[: max(requested * 4, requested)]]


def _collect_series(path: Path, keys: Sequence[Key]) -> Dict[Key, List[SeriesRow]]:
    wanted = set(keys)
    result: MutableMapping[Key, List[SeriesRow]] = defaultdict(list)
    for row in _movement_rows(path):
        if _int(row, "OK") != 1:
            continue
        key = (_int(row, "STORE", -1), _int(row, "UPC", -1))
        if key not in wanted:
            continue
        qty = _float(row, "QTY")
        bundle_price = _float(row, "PRICE")
        unit_price = bundle_price / qty if qty > 0.0 and bundle_price > 0.0 else 0.0
        profit = _float(row, "PROFIT", float("nan"))
        result[key].append(
            SeriesRow(
                week=_int(row, "WEEK"),
                demand=max(0.0, _float(row, "MOVE")),
                price=max(0.0, unit_price),
                sale=bool(str(row.get("SALE", "")).strip()),
                profit_pct=profit if math.isfinite(profit) else None,
            )
        )
    return {key: sorted(rows, key=lambda item: item.week) for key, rows in result.items()}


def _observed_price_rows(rows: Sequence[SeriesRow]) -> Dict[int, SeriesRow]:
    """Keep only weeks with an observed positive historical unit price.

    A zero-price record can reflect stockout, delisting, assortment change, or
    another availability event.  Carrying a prior price into such a record
    would turn that event into an artificial demand residual and would use a
    future reappearance to decide whether the week belongs to the active
    sample.  Availability is therefore measured separately from demand-shock
    coverage.
    """
    return {row.week: row for row in rows if row.price > 0.0}


def _spaced_menu(prices: Sequence[float], maximum: int) -> np.ndarray:
    values = np.array(sorted(set(round(float(price), 2) for price in prices)), dtype=float)
    if values.size <= maximum:
        return values
    indices = np.unique(np.rint(np.linspace(0, values.size - 1, maximum)).astype(int))
    return values[indices]


def _repeated_training_menu(
    prices: Sequence[float], maximum: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return only rounded training prices observed at least twice."""
    counts = Counter(round(float(value), 2) for value in prices)
    repeated = [price for price, count in counts.items() if count >= 2]
    menu = _spaced_menu(repeated, maximum)
    occurrences = np.array(
        [counts[round(float(price), 2)] for price in menu], dtype=int
    )
    return menu, occurrences


def _fit_item(
    key: Key,
    rows: Dict[int, SeriesRow],
    split: Tuple[int, int, int],
    band: float,
    max_menu: int,
) -> ItemModel | None:
    train_end, tune_end, _ = split
    train = [row for week, row in rows.items() if week <= train_end]
    tune = [row for week, row in rows.items() if train_end < week <= tune_end]
    if len(train) < 80:
        return None
    X = np.vstack([_features(row.price, row.week, row.sale) for row in train])
    y = np.log1p(np.array([row.demand for row in train], dtype=float))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    elasticity = float(beta[1])
    if not math.isfinite(elasticity) or elasticity >= -0.02:
        return None

    regular_prices = [row.price for row in train if not row.sale and row.price > 0.0]
    if len(regular_prices) < 20:
        regular_prices = [row.price for row in train if row.price > 0.0]
    reference = float(statistics.median(regular_prices))
    admissible = [
        row.price
        for row in train
        if (
            row.price > 0.0
            and not row.sale
            and abs(row.price - reference) <= band * reference + 1e-12
        )
    ]
    menu, menu_occurrences = _repeated_training_menu(admissible, max_menu)
    if menu.size < 3:
        return None

    cost_observations = []
    for row in train:
        if row.profit_pct is None or not (-20.0 <= row.profit_pct < 100.0):
            continue
        cost_observations.append(row.price * (1.0 - row.profit_pct / 100.0))
    if len(cost_observations) < 20:
        return None
    cost = float(statistics.median(cost_observations))
    evaluation_week = tune_end
    demand = np.array(
        [
            max(1e-6, math.expm1(_features(p, evaluation_week, False) @ beta))
            for p in menu
        ]
    )

    return ItemModel(
        key=key,
        beta=beta,
        cost=cost,
        reference=reference,
        menu=menu,
        demand=demand,
        rows=rows,
        train_rows=len(train),
        tune_rows=len(tune),
        log_price_coefficient=elasticity,
        baseline_week=evaluation_week,
        menu_training_regular_occurrences=menu_occurrences,
    )


def _bounded_shock_vectors(
    models: Sequence[ItemModel],
    start_exclusive: int,
    end_inclusive: int,
) -> Tuple[List[int], np.ndarray]:
    common = set(range(start_exclusive + 1, end_inclusive + 1))
    for model in models:
        common &= set(model.rows)
    weeks = sorted(common)
    vectors = []
    for week in weeks:
        eps = []
        for model in models:
            row = model.rows[week]
            prediction = model.predict(row.price, week, row.sale)
            if row.demand <= prediction:
                # Downward relative shortfall, including -1 for zero demand.
                eps.append(-(prediction - row.demand) / prediction)
            else:
                # Bounded transform of an upward relative error.  Its inverse
                # is demand / prediction = 1 / (1 - eps).
                eps.append((row.demand - prediction) / row.demand)
        vectors.append(eps)
    return weeks, np.asarray(vectors, dtype=float)


def _positive_price_availability(
    models: Sequence[ItemModel],
    start_exclusive: int,
    end_inclusive: int,
) -> Tuple[List[int], List[int]]:
    weeks = list(range(start_exclusive + 1, end_inclusive + 1))
    counts = [sum(week in model.rows for model in models) for week in weeks]
    return weeks, counts


def _conformal_radius(scores: Sequence[float], alpha: float) -> float:
    ordered = sorted(float(score) for score in scores)
    k = int(math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    if k > len(ordered):
        return float("inf")
    return ordered[k - 1]


def _build_instance(
    models: Sequence[ItemModel],
    gamma: int,
    radius: float,
    kappa: float,
) -> PricingInstance:
    if not (0.0 <= radius < 1.0):
        raise ValueError("bounded-shock radius must lie in [0,1)")
    downward_factor = radius
    upward_factor = radius / (1.0 - radius) if radius > 0.0 else 0.0
    items: List[List[Option]] = []
    for model in models:
        group = []
        for price, demand in zip(model.menu, model.demand):
            value = price * demand
            unit_surplus = price - kappa * model.cost
            # A demand decrease is adverse for a nonnegative unit surplus;
            # a demand increase is adverse for a negative unit surplus.  The
            # bounded residual transform yields the two safe endpoint widths.
            adverse_factor = downward_factor if unit_surplus >= 0.0 else upward_factor
            group.append(
                Option(
                    value=float(value),
                    margin=float(unit_surplus * demand),
                    uncertainty=float(abs(unit_surplus) * adverse_factor * demand),
                    price=float(price),
                )
            )
        items.append(group)
    return PricingInstance(
        items=items, gamma=int(gamma), name="dominicks-cereal-v3-pilot"
    )


def _evaluate_policy(
    models: Sequence[ItemModel],
    selections: Sequence[int],
    weeks: Sequence[int],
    residuals: np.ndarray,
    kappa: float,
) -> dict:
    """Apply held-out shock scenarios to the fixed encoded demand vector.

    The pricing instance and its selected price vector use ``model.demand``,
    evaluated at the training-frozen baseline week.  Historical weeks supply
    only bounded residual scenarios here; candidate-price demand is not
    re-forecast with the test week's seasonal covariates.
    """
    ratios: List[float] = []
    revenues: List[float] = []
    surpluses: List[float] = []
    for row_idx, _week in enumerate(weeks):
        revenue = 0.0
        cost = 0.0
        for i, model in enumerate(models):
            j = int(selections[i])
            shock = float(residuals[row_idx, i])
            if shock <= 0.0:
                realized = max(0.0, model.demand[j] * (1.0 + shock))
            else:
                realized = model.demand[j] / max(1.0 - shock, 1e-12)
            revenue += float(model.menu[j]) * realized
            cost += model.cost * realized
        revenues.append(revenue)
        ratios.append(revenue / cost if cost > 0.0 else float("inf"))
        surpluses.append(revenue - kappa * cost)
    return {
        "weeks": len(weeks),
        "violation_rate": float(np.mean(np.asarray(ratios) < kappa)) if ratios else float("nan"),
        "mean_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "mean_revenue": float(np.mean(revenues)) if revenues else float("nan"),
        "mean_surplus": float(np.mean(surpluses)) if surpluses else float("nan"),
        "p05_surplus": _percentile(surpluses, 0.05),
        "min_surplus": min(surpluses) if surpluses else float("nan"),
    }


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def run(args: argparse.Namespace) -> dict:
    calendar_start, calendar_end = _calendar_weeks(args.movement_zip)
    calendar_length = calendar_end - calendar_start + 1
    boundaries = tuple(
        calendar_start + min(calendar_length - 1, int(q * calendar_length))
        for q in (0.50, 0.65, 0.80)
    )
    stats = _first_pass(args.movement_zip, boundaries[0])
    store, candidate_keys = _choose_keys(stats, args.n_items)
    raw_series = _collect_series(args.movement_zip, candidate_keys)
    observed = {
        key: _observed_price_rows(rows) for key, rows in raw_series.items()
    }

    fitted = []
    candidates_attempted = 0
    for key in candidate_keys:
        candidates_attempted += 1
        model = _fit_item(
            key, observed.get(key, {}), boundaries, args.band, args.max_menu
        )
        if model is not None:
            fitted.append(model)
        if len(fitted) >= args.n_items:
            break
    if len(fitted) < max(5, min(args.n_items, 8)):
        raise RuntimeError(f"only {len(fitted)} items passed predictive/menu filters")

    train_end, tune_end, calibration_end = boundaries
    tune_weeks, tune_eps = _bounded_shock_vectors(fitted, train_end, tune_end)
    cal_weeks, cal_eps = _bounded_shock_vectors(fitted, tune_end, calibration_end)
    test_weeks, test_eps = _bounded_shock_vectors(fitted, calibration_end, calendar_end)
    if min(len(tune_weeks), len(cal_weeks)) < 10:
        raise RuntimeError(
            "fewer than ten complete positive-price portfolio weeks in tuning "
            "or calibration"
        )
    if not test_weeks:
        raise RuntimeError("no complete positive-price portfolio week in the test block")

    availability_weeks, availability_counts = _positive_price_availability(
        fitted, tune_end, calendar_end
    )

    tune_budget_mass = np.sum(np.abs(tune_eps), axis=1)
    gamma = int(max(1, min(len(fitted), math.ceil(_percentile(tune_budget_mass.tolist(), 0.90)))))
    cal_scores = np.maximum(np.max(np.abs(cal_eps), axis=1), np.sum(np.abs(cal_eps), axis=1) / gamma)
    radius = _conformal_radius(cal_scores.tolist(), args.alpha)
    if not math.isfinite(radius):
        raise RuntimeError("calibration block too small for the requested conformal level")
    if radius >= 1.0:
        raise RuntimeError(
            "the calibrated bounded-shock radius is at least one; the asymmetric "
            "upper-demand translation is therefore unbounded"
        )

    empirical_coverage = float(np.mean(cal_scores <= radius))
    test_scores = np.maximum(
        np.max(np.abs(test_eps), axis=1),
        np.sum(np.abs(test_eps), axis=1) / gamma,
    )
    rolling_window = len(cal_scores)
    rolling_history = [float(value) for value in cal_scores]
    rolling_radii: List[float] = []
    rolling_hits: List[bool] = []
    for score in test_scores:
        rolling_radius = _conformal_radius(
            rolling_history[-rolling_window:], args.alpha
        )
        rolling_radii.append(rolling_radius)
        rolling_hits.append(float(score) <= rolling_radius)
        rolling_history.append(float(score))
    target_results: Dict[str, dict] = {}
    certificate_directory = args.output.parent / f"{args.output.stem}_certificates"
    certificate_directory.mkdir(parents=True, exist_ok=True)
    for loss_ratio_cap in args.loss_ratio_caps:
        kappa = 1.0 / loss_ratio_cap
        robust_instance = _build_instance(fitted, gamma, radius, kappa)
        nominal_instance = _build_instance(fitted, 0, radius, kappa)
        robust_solution = solve(robust_instance)
        nominal_solution = solve(nominal_instance)
        target_key = f"{loss_ratio_cap:.4f}"
        target: Dict[str, object] = {
            "loss_ratio_cap": loss_ratio_cap,
            "kappa": kappa,
            "analysis_role": (
                "original_exploratory_target"
                if abs(loss_ratio_cap - 0.80) <= 1e-12
                else "post_hoc_exploratory_sensitivity"
            ),
            "nominal_feasible": nominal_solution.is_feasible,
            "robust_feasible": robust_solution.is_feasible,
        }
        if nominal_solution.is_feasible:
            nominal_eval = _evaluate_policy(
                fitted, nominal_solution.selections, test_weeks, test_eps, kappa
            )
            target["nominal"] = {
                "objective": nominal_solution.objective,
                "selections": nominal_solution.selections,
                "prices": [
                    float(model.menu[j])
                    for model, j in zip(fitted, nominal_solution.selections)
                ],
                "certificate_residual": compute_certificate(
                    nominal_instance, nominal_solution.selections
                ),
                "test": {
                    key: _json_number(value) if isinstance(value, float) else value
                    for key, value in nominal_eval.items()
                },
            }
        else:
            target["nominal_message"] = nominal_solution.metadata
        if robust_solution.is_feasible:
            robust_eval = _evaluate_policy(
                fitted, robust_solution.selections, test_weeks, test_eps, kappa
            )
            target["robust"] = {
                "objective": robust_solution.objective,
                "selections": robust_solution.selections,
                "prices": [
                    float(model.menu[j])
                    for model, j in zip(fitted, robust_solution.selections)
                ],
                "certificate_residual": compute_certificate(
                    robust_instance, robust_solution.selections
                ),
                "minimum_selected_unit_surplus": min(
                    float(model.menu[j] - kappa * model.cost)
                    for model, j in zip(fitted, robust_solution.selections)
                ),
                "all_selected_unit_surpluses_nonnegative": all(
                    float(model.menu[j] - kappa * model.cost) >= 0.0
                    for model, j in zip(fitted, robust_solution.selections)
                ),
                "test": {
                    key: _json_number(value) if isinstance(value, float) else value
                    for key, value in robust_eval.items()
                },
            }
            global_certificate = build_global_certificate(
                robust_instance, robust_solution.selections
            )
            certificate_raw = global_certificate.to_json(indent=2) + "\n"
            certificate_check = verify_global_certificate(
                robust_instance, certificate_raw
            )
            if not certificate_check.valid:
                raise RuntimeError(
                    "exact global certificate failed for loss-ratio cap "
                    f"{loss_ratio_cap}: {'; '.join(certificate_check.errors)}"
                )
            certificate_path = certificate_directory / (
                f"loss-ratio-{loss_ratio_cap:.4f}_robust.json"
            )
            certificate_path.write_text(certificate_raw, encoding="utf-8")
            target["robust"]["global_certificate"] = {
                "file": certificate_path.name,
                "sha256": _sha256(certificate_path),
                "verified": True,
                "H": str(global_certificate.h),
                "U": str(global_certificate.u),
                "absolute_gap": str(global_certificate.absolute_gap),
                "relative_width": (
                    None
                    if global_certificate.relative_width is None
                    else str(global_certificate.relative_width)
                ),
            }
            if nominal_solution.is_feasible and nominal_solution.objective > 0.0:
                target["nominal_to_robust_objective_change"] = (
                    robust_solution.objective / nominal_solution.objective - 1.0
                )
        else:
            target["robust_message"] = robust_solution.metadata
        target_results[target_key] = target

    availability_gate_passed = len(test_weeks) >= 10
    reliability_reasons = [
        "serial dependence and nonstationarity cannot be ruled out and are "
        "not modeled"
    ]
    if not availability_gate_passed:
        reliability_reasons.insert(
            0,
            "fewer than ten complete positive-price portfolio weeks in test",
        )

    result = {
        "status": (
            "descriptive_only_unmodeled_dependence"
            if availability_gate_passed
            else "failed_availability_and_unmodeled_dependence_gate"
        ),
        "interpretation": (
            "predictive semi-synthetic bounded-common-shock pilot; not a causal "
            "deployment estimate"
        ),
        "protocol_role": (
            "post-hoc exploratory chronological stress test; split boundaries, "
            "filters, and the availability gate were revised after the future "
            "block had been inspected; not preregistered or an untouched test"
        ),
        "data": {
            "movement_file": args.movement_zip.name,
            "movement_logical_source": "Dominicks/cereals/wcer.zip",
            "movement_sha256": _sha256(args.movement_zip),
            "upc_file": args.upc_csv.name,
            "upc_logical_source": "Dominicks/cereals/upccer.csv",
            "upc_sha256": _sha256(args.upc_csv),
            "upc_use": (
                "downloaded and hashed for dataset provenance; not read by the "
                "pilot computation"
            ),
            "store": store,
            "store_selection_rule": (
                "training-period store with the largest eligible-UPC count; "
                "ties broken by larger store identifier"
            ),
            "store_representativeness": "deliberately selected; not representative",
            "candidate_keys": len(candidate_keys),
            "candidate_ranking_rule": (
                "descending training positive-price records, distinct regular "
                "prices, movement, and UPC identifier"
            ),
            "candidates_attempted_until_roster_filled": candidates_attempted,
            "candidate_filter_rejections_before_roster_filled": (
                candidates_attempted - len(fitted)
            ),
            "modeled_items": len(fitted),
            "week_boundaries": {
                "train_end": train_end,
                "tune_end": tune_end,
                "calibration_end": calibration_end,
                "test_end": calendar_end,
            },
        },
        "calibration": {
            "alpha": args.alpha,
            "gamma": gamma,
            "radius": radius,
            "tune_complete_weeks": len(tune_weeks),
            "calibration_complete_weeks": len(cal_weeks),
            "test_complete_weeks": len(test_weeks),
            "calibration_empirical_coverage": empirical_coverage,
            "test_empirical_coverage": float(np.mean(test_scores <= radius)),
            "test_coordinate_coverage": float(
                np.mean(np.max(np.abs(test_eps), axis=1) <= radius)
            ),
            "test_budget_coverage": float(
                np.mean(np.sum(np.abs(test_eps), axis=1) <= gamma * radius)
            ),
            "test_score_maximum": float(np.max(test_scores)),
            "rolling_window_weeks": rolling_window,
            "rolling_one_step_empirical_coverage": float(np.mean(rolling_hits)),
            "rolling_finite_width_fraction": float(
                np.mean(np.asarray(rolling_radii) < 1.0)
            ),
            "rolling_radius_median": float(np.median(rolling_radii)),
            "rolling_radius_maximum": float(np.max(rolling_radii)),
            "bounded_shock_definition": (
                "-(prediction-demand)/prediction below prediction; "
                "+(demand-prediction)/demand above prediction"
            ),
            "upward_adverse_width_factor": radius / (1.0 - radius),
            "calibration_calendar_weeks": calibration_end - tune_end,
            "test_calendar_weeks": calendar_end - calibration_end,
            "calibration_complete_positive_price_fraction": (
                len(cal_weeks) / (calibration_end - tune_end)
            ),
            "test_complete_positive_price_fraction": (
                len(test_weeks) / (calendar_end - calibration_end)
            ),
            "complete_positive_price_test_week_numbers": test_weeks,
            "calibration_week_numbers": cal_weeks,
            "calibration_scores": [float(value) for value in cal_scores],
            "test_week_numbers": test_weeks,
            "test_scores": [float(value) for value in test_scores],
            "availability_week_numbers": availability_weeks,
            "positive_price_item_counts": availability_counts,
            "availability_gate_passed": availability_gate_passed,
            "reliability_gate_passed": False,
            "reliability_gate_reasons": reliability_reasons,
            "availability_interpretation": (
                "zero-price records are availability events, not imputed demand "
                "residuals; coverage is evaluated only on complete positive-price "
                "portfolio weeks"
            ),
        },
        "model": {
            "loss_ratio_caps": args.loss_ratio_caps,
            "price_band": args.band,
            "menu_rule": (
                "each menu price appears at least twice without a sale flag in "
                "training and lies within the price band"
            ),
            "nominal_demand_baseline_week": tune_end,
            "test_scenario_construction": (
                "held-out bounded residuals are applied to the fixed baseline-week "
                "candidate-price demand vector; no week-specific candidate-price "
                "reforecast is made"
            ),
            "menu_sizes": [int(model.menu.size) for model in fitted],
            "log_price_coefficients": [
                model.log_price_coefficient for model in fitted
            ],
            "items": [
                {
                    "store": model.key[0],
                    "upc": model.key[1],
                    "reference_price": model.reference,
                    "accounting_unit_cost": model.cost,
                    "menu": model.menu.tolist(),
                    "menu_training_regular_occurrences": (
                        model.menu_training_regular_occurrences.tolist()
                    ),
                    "demand": model.demand.tolist(),
                    "demand_baseline_week": model.baseline_week,
                }
                for model in fitted
            ],
        },
        "targets": target_results,
    }
    for target in result["targets"].values():
        for policy_name in ("nominal", "robust"):
            policy = target.get(policy_name)
            if not isinstance(policy, dict):
                continue
            test = policy["test"]
            total = int(test["weeks"])
            violations = int(round(float(test["violation_rate"]) * total))
            test["violation_count"] = violations
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movement-zip", type=Path, required=True)
    parser.add_argument("--upc-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-items", type=int, default=12)
    parser.add_argument("--max-menu", type=int, default=8)
    parser.add_argument("--band", type=float, default=0.10)
    parser.add_argument(
        "--loss-ratio-caps",
        type=float,
        nargs="+",
        default=[0.80, 0.85, 0.90, 0.95],
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.loss_ratio_caps or any(
        not (0.0 < cap < 1.0) for cap in args.loss_ratio_caps
    ):
        raise ValueError("every loss-ratio cap must lie in (0,1)")
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("alpha must lie in (0,1)")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
