#!/usr/bin/env python3
"""Ablate fixed-θ data construction and root-bound maintenance."""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_mckp import Option, PricingInstance  # noqa: E402
from robust_mckp.exact_bnb import (  # noqa: E402
    _build_fixed_theta_cache,
    build_fixed_theta_data,
    build_full_theta_candidates,
    compute_fixed_theta_lp_upper_bound,
)
from robust_mckp.parametric_sweep import ParametricThetaSweepConfig, build_parametric_theta_sweep  # noqa: E402


CSV_NAME = "ablation_results.csv"
DEFAULT_METHODS = ["independent_recompute", "incremental_sweep_rebuild", "incremental_sweep_safe_reuse"]


def _gamma_sqrt(n: int) -> int:
    return int(math.floor(math.sqrt(n)))


def parse_csv_arg(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def make_instance(family: str, n: int, m: int, gamma: int, seed: int) -> PricingInstance:
    rng = np.random.default_rng(seed + 1009 * n + 9173 * m + 7919 * gamma)
    items = []
    for i in range(n):
        group = []
        for j in range(m):
            value = float(max(0, rng.normal(20.0 + 1.5 * j, 5.0)))
            if family == "many_theta":
                margin = float(10.0 - 0.45 * j + rng.normal(0.0, 1.0))
                uncertainty = float((i * m + j + 1) / (n * m) * max(2.0, 0.35 * n))
            elif family == "tight_capacity":
                margin = float(3.5 - 0.7 * j + rng.normal(0.0, 0.6))
                uncertainty = float(rng.integers(0, max(2, min(8, n // 3 + 2))))
            elif family in {"low_compression", "adversarial"}:
                margin = float(8.0 - 0.55 * j + rng.normal(0.0, 1.5))
                uncertainty = float((j + 1) * (1.0 + 0.15 * (i % 7)) + rng.normal(0.0, 0.08))
            elif family in {"hull_compression", "non_tight_control"}:
                margin = float(14.0 - 0.3 * j + rng.normal(0.0, 1.0))
                uncertainty = float(rng.integers(0, 4))
            else:
                margin = float(8.0 - 0.5 * j + rng.normal(0.0, 1.0))
                uncertainty = float(rng.integers(0, 8))
            group.append(Option(value=value, margin=margin, uncertainty=uncertainty))
        group[0] = Option(value=float(max(1.0, rng.normal(8.0, 2.0))), margin=max(4.0, group[0].margin), uncertainty=0.0)
        items.append(group)
    return PricingInstance(items=items, gamma=gamma, name=f"{family}_n{n}_m{m}_g{gamma}_s{seed}")


def grid(args: argparse.Namespace) -> Iterable[tuple[str, int, int, int, int]]:
    families = parse_csv_arg(args.families)
    if args.smoke:
        for family in families:
            n = 20
            yield family, n, 6, _gamma_sqrt(n), 0
        return
    if args.seeds is not None:
        seeds = list(range(int(args.seeds)))
    elif args.paper_full:
        seeds = [0, 1, 2]
    else:
        seeds = [0, 1]
    if args.paper_full:
        ns = [50, 80, 100, 150, 200]
        ms = [8, 10, 12]
    else:
        ns = [50, 80, 100]
        ms = [8, 10]
    emitted = 0
    for family in families:
        for n in ns:
            for m in ms:
                for gamma in sorted(set([_gamma_sqrt(n), int(math.floor(0.1 * n))])):
                    for seed in seeds:
                        if args.max_instances is not None and emitted >= args.max_instances:
                            return
                        emitted += 1
                        yield family, n, m, gamma, seed


def base_row(family: str, n: int, m: int, gamma: int, seed: int, method: str) -> Dict[str, object]:
    return {
        "family": family,
        "n": n,
        "m": m,
        "gamma": gamma,
        "seed": seed,
        "method": method,
        "status": "not_run",
        "theta_count_total": 0,
        "theta_candidate_source": "full_original_breakpoints",
        "root_bounds_computed": 0,
        "total_runtime_seconds": 0.0,
        "candidate_generation_time_seconds": 0.0,
        "theta_data_time_seconds": 0.0,
        "baseline_cost_capacity_time_seconds": 0.0,
        "hull_time_seconds": 0.0,
        "root_lp_time_seconds": 0.0,
        "hull_rebuilds_total": 0,
        "hull_reuses_total": 0,
        "item_hulls_changed_total": 0,
        "item_hulls_unchanged_total": 0,
        "dirty_item_fraction": 0.0,
        "hull_reuse_rate": 0.0,
        "max_abs_s_theta_recompute_error": 0.0,
        "max_abs_cost_recompute_error": 0.0,
        "max_abs_capacity_recompute_error": 0.0,
        "max_abs_root_bound_recompute_error": 0.0,
        "error_message": "",
    }


def run_independent(instance: PricingInstance, row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    start = time.perf_counter()
    theta_data_time = 0.0
    candidate_time = 0.0
    hull_time = 0.0
    root_time = 0.0
    computed = 0
    t0 = time.perf_counter()
    candidates = build_full_theta_candidates(instance)
    candidate_time = time.perf_counter() - t0
    try:
        for theta in candidates:
            t0 = time.perf_counter()
            data = build_fixed_theta_data(instance, theta)
            theta_data_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            cache = _build_fixed_theta_cache(instance, theta, 1e-9, data=data)
            hull_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            compute_fixed_theta_lp_upper_bound(instance, theta, 1e-9, cache=cache)
            root_time += time.perf_counter() - t0
            computed += 1
        out.update(
            {
                "status": "ok",
                "theta_count_total": len(candidates),
                "theta_candidate_source": "full_original_breakpoints",
                "root_bounds_computed": computed,
                "total_runtime_seconds": time.perf_counter() - start,
                "candidate_generation_time_seconds": candidate_time,
                "theta_data_time_seconds": theta_data_time,
                "baseline_cost_capacity_time_seconds": theta_data_time,
                "hull_time_seconds": hull_time,
                "root_lp_time_seconds": root_time,
                "hull_rebuilds_total": len(candidates) * instance.n_items,
                "item_hulls_changed_total": len(candidates) * instance.n_items,
                "item_hulls_unchanged_total": 0,
                "dirty_item_fraction": 1.0,
            }
        )
    except Exception as exc:
        out.update({"status": "error", "total_runtime_seconds": time.perf_counter() - start, "error_message": str(exc)})
    return out


def run_sweep(instance: PricingInstance, row: Dict[str, object], *, reuse: bool, validate: bool, sampled: bool) -> Dict[str, object]:
    out = dict(row)
    start = time.perf_counter()
    try:
        config = ParametricThetaSweepConfig(
            validate_against_recompute=validate or sampled,
            reuse_hulls=reuse,
            force_rebuild_hulls=not reuse,
            max_recompute_checks=10 if sampled and not validate else None,
        )
        result = build_parametric_theta_sweep(instance, config=config)
        diag = result.diagnostics
        out.update(
            {
                "status": "ok",
                "theta_count_total": diag["theta_count_total"],
                "theta_candidate_source": diag.get("theta_candidate_source", "full_original_breakpoints"),
                "root_bounds_computed": diag["theta_count_total"],
                "total_runtime_seconds": time.perf_counter() - start,
                "candidate_generation_time_seconds": diag.get("candidate_generation_time_seconds", 0.0),
                "theta_data_time_seconds": diag.get("sweep_update_time_seconds", 0.0),
                "baseline_cost_capacity_time_seconds": diag.get("baseline_update_time_seconds", 0.0),
                "hull_time_seconds": sum(float(r.hull_time_seconds) for r in result.records),
                "root_lp_time_seconds": diag.get("root_lp_time_seconds", 0.0),
                "hull_rebuilds_total": diag.get("hull_rebuilds_total", 0),
                "hull_reuses_total": diag.get("hull_reuses_total", 0),
                "item_hulls_changed_total": diag.get("item_hulls_changed_total", 0),
                "item_hulls_unchanged_total": diag.get("item_hulls_unchanged_total", 0),
                "dirty_item_fraction": diag.get("dirty_item_fraction", 0.0),
                "hull_reuse_rate": diag.get("hull_reuse_rate", 0.0),
                "max_abs_s_theta_recompute_error": diag.get("max_abs_s_theta_recompute_error", 0.0),
                "max_abs_cost_recompute_error": diag.get("max_abs_cost_recompute_error", 0.0),
                "max_abs_capacity_recompute_error": diag.get("max_abs_capacity_recompute_error", 0.0),
                "max_abs_root_bound_recompute_error": diag.get("max_abs_root_lp_recompute_error", 0.0),
            }
        )
    except Exception as exc:
        out.update({"status": "error", "total_runtime_seconds": time.perf_counter() - start, "error_message": str(exc)})
    return out


def row_key(row: Dict[str, object]) -> tuple[object, ...]:
    return (row["family"], row["n"], row["m"], row["gamma"], row["seed"], row["method"])


def load_existing(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> List[Dict[str, object]]:
    output_path = args.output_dir / CSV_NAME
    rows: List[Dict[str, object]] = load_existing(output_path) if args.resume else []
    done = {row_key(r) for r in rows}
    methods = parse_csv_arg(args.methods)
    cases = list(grid(args))
    for idx, (family, n, m, gamma, seed) in enumerate(cases, start=1):
        instance = make_instance(family, n, m, gamma, seed)
        print(f"[{idx}/{len(cases)}] {family} n={n} m={m} gamma={gamma} seed={seed}", flush=True)
        for method in methods:
            row = base_row(family, n, m, gamma, seed, method)
            if row_key(row) in done:
                continue
            if method == "independent_recompute":
                result = run_independent(instance, row)
            elif method == "incremental_sweep_rebuild":
                result = run_sweep(instance, row, reuse=False, validate=args.validate, sampled=args.validate_sampled)
            elif method == "incremental_sweep_safe_reuse":
                result = run_sweep(instance, row, reuse=True, validate=args.validate, sampled=args.validate_sampled)
            elif method == "validation_mode":
                result = run_sweep(instance, row, reuse=True, validate=True, sampled=False)
            else:
                result = dict(row)
                result.update({"status": "error", "error_message": f"unknown method {method}"})
            rows.append(result)
            done.add(row_key(result))
            write_rows(rows, output_path)
            if args.fail_fast and result["status"] == "error":
                raise RuntimeError(str(result["error_message"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--paper-lite", action="store_true")
    parser.add_argument("--paper-full", action="store_true")
    parser.add_argument("--families", default="many_theta,tight_capacity,non_tight_control")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "parametric_sweep_ablation")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-sampled", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    run(args)
    print(f"Wrote {args.output_dir / CSV_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
