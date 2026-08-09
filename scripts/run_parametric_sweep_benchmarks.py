#!/usr/bin/env python3
"""Run exact enumeration vs exact sweep solver benchmarks."""
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

from robust_mckp import GlobalThetaBNBConfig, Option, PricingInstance, solve  # noqa: E402
from robust_mckp.certificate import certificate_is_feasible  # noqa: E402
from robust_mckp.exact_bnb import solve_global_theta_bnb  # noqa: E402
from robust_mckp.milp_baselines import solve_theta_decomposition_milp_baseline  # noqa: E402
from robust_mckp.parametric_sweep import ParametricThetaSweepConfig, solve_global_theta_bnb_sweep  # noqa: E402


CSV_NAME = "parametric_sweep_results.csv"
DEFAULT_METHODS = [
    "hullround",
    "exact_enum_current",
    "exact_sweep_new",
    "exact_sweep_new_validation_sampled",
    "scipy_highs",
    "scip",
    "gurobi",
    "cplex",
]


def parse_csv_arg(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _gamma_sqrt(n: int) -> int:
    return int(math.floor(math.sqrt(n)))


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
            else:
                margin = float(14.0 - 0.3 * j + rng.normal(0.0, 1.0))
                uncertainty = float(rng.integers(0, 4))
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


def base_row(family: str, n: int, m: int, gamma: int, seed: int, method: str, theta_order: str) -> Dict[str, object]:
    return {
        "family": family,
        "n": n,
        "m": m,
        "gamma": gamma,
        "seed": seed,
        "method": method,
        "status": "not_run",
        "certified_optimal": False,
        "feasible_incumbent": False,
        "objective_value": float("nan"),
        "upper_bound": float("nan"),
        "lower_bound": float("nan"),
        "abs_gap": float("nan"),
        "rel_gap": float("nan"),
        "runtime_seconds": float("nan"),
        "theta_order": theta_order,
        "theta_candidate_source": "full_original_breakpoints",
        "theta_count_total": 0,
        "theta_count_evaluated": 0,
        "theta_count_pruned": 0,
        "theta_count_solved_bnb": 0,
        "theta_count_limited_bnb": 0,
        "total_nodes_explored": 0,
        "root_lp_time_seconds": 0.0,
        "fixed_theta_bnb_time_seconds": 0.0,
        "candidate_generation_time_seconds": 0.0,
        "sweep_update_time_seconds": 0.0,
        "theta_data_time_seconds": 0.0,
        "baseline_cost_capacity_time_seconds": 0.0,
        "hull_rebuilds_total": 0,
        "hull_reuses_total": 0,
        "item_hulls_changed_total": 0,
        "item_hulls_unchanged_total": 0,
        "dirty_item_fraction": 0.0,
        "hull_reuse_rate": 0.0,
        "certificate_validation_time_seconds": 0.0,
        "certificate_validation_count": 0,
        "prune_count_by_reason": "",
        "validation_mode": "none",
        "max_abs_validation_error": 0.0,
        "robust_certificate": float("nan"),
        "robust_certificate_passed": False,
        "backend_available": True,
        "backend_message": "",
    }


def row_from_global(base: Dict[str, object], result) -> Dict[str, object]:
    diag = result.diagnostics or {}
    max_validation = max(
        float(diag.get("max_abs_s_theta_recompute_error", 0.0) or 0.0),
        float(diag.get("max_abs_cost_recompute_error", 0.0) or 0.0),
        float(diag.get("max_abs_capacity_recompute_error", 0.0) or 0.0),
    )
    row = dict(base)
    row.update(
        {
            "status": result.status,
            "certified_optimal": result.status == "optimal",
            "feasible_incumbent": bool(result.validation_flags.get("robust_certificate_feasible", False)),
            "objective_value": result.objective_value,
            "upper_bound": result.upper_bound,
            "lower_bound": result.lower_bound,
            "abs_gap": result.absolute_gap,
            "rel_gap": result.relative_gap,
            "runtime_seconds": result.total_runtime_seconds,
            "theta_candidate_source": diag.get("theta_candidate_source", "full_original_breakpoints"),
            "theta_count_total": result.theta_count_total,
            "theta_count_evaluated": result.theta_count_total - result.theta_count_infeasible_capacity,
            "theta_count_pruned": result.theta_count_pruned_by_bound,
            "theta_count_solved_bnb": result.theta_count_solved_optimal,
            "theta_count_limited_bnb": result.theta_count_limited,
            "total_nodes_explored": result.total_nodes_explored,
            "root_lp_time_seconds": result.total_root_lp_time_seconds,
            "fixed_theta_bnb_time_seconds": result.total_fixed_theta_bnb_time_seconds,
            "candidate_generation_time_seconds": diag.get("candidate_generation_time_seconds", 0.0),
            "sweep_update_time_seconds": diag.get("sweep_update_time_seconds", 0.0),
            "theta_data_time_seconds": diag.get("sweep_update_time_seconds", 0.0),
            "baseline_cost_capacity_time_seconds": diag.get("baseline_update_time_seconds", 0.0),
            "hull_rebuilds_total": diag.get("hull_rebuilds_total", 0),
            "hull_reuses_total": diag.get("hull_reuses_total", 0),
            "item_hulls_changed_total": diag.get("item_hulls_changed_total", 0),
            "item_hulls_unchanged_total": diag.get("item_hulls_unchanged_total", 0),
            "dirty_item_fraction": diag.get("dirty_item_fraction", 0.0),
            "hull_reuse_rate": diag.get("hull_reuse_rate", 0.0),
            "certificate_validation_time_seconds": diag.get("certificate_validation_time_seconds", 0.0),
            "certificate_validation_count": diag.get("certificate_validation_count", 0),
            "prune_count_by_reason": diag.get("prune_count_by_reason", ""),
            "validation_mode": diag.get("validation_mode", "none"),
            "max_abs_validation_error": max_validation,
            "robust_certificate": result.robust_certificate,
            "robust_certificate_passed": bool(result.validation_flags.get("robust_certificate_feasible", False)),
        }
    )
    return row


def run_hullround(instance: PricingInstance, base: Dict[str, object]) -> Dict[str, object]:
    row = dict(base)
    t0 = time.perf_counter()
    try:
        sol = solve(instance)
        instr = (sol.metadata or {}).get("instrumentation", {})
        row.update(
            {
                "status": "feasible" if sol.is_feasible else "infeasible",
                "feasible_incumbent": sol.is_feasible,
                "objective_value": sol.objective,
                "lower_bound": sol.objective,
                "upper_bound": sol.lp_value,
                "abs_gap": max(0.0, sol.lp_value - sol.objective) if math.isfinite(sol.lp_value) else float("nan"),
                "rel_gap": sol.gap_to_lp,
                "runtime_seconds": time.perf_counter() - t0,
                "theta_count_total": instr.get("theta_evaluated_count", 0),
                "theta_count_evaluated": instr.get("theta_evaluated_count", 0),
                "hull_rebuilds_total": instr.get("total_hull_rebuilds", 0),
                "robust_certificate": sol.certificate_value,
                "robust_certificate_passed": bool(sol.is_feasible),
            }
        )
    except Exception as exc:
        row.update({"status": "error", "runtime_seconds": time.perf_counter() - t0, "backend_message": str(exc)})
    return row


def run_optional_baseline(instance: PricingInstance, backend: str, base: Dict[str, object], time_limit: float) -> Dict[str, object]:
    result = solve_theta_decomposition_milp_baseline(instance, backend=backend, time_limit_per_theta=time_limit)
    row = dict(base)
    row.update(
        {
            "status": result.status,
            "certified_optimal": result.status == "optimal",
            "feasible_incumbent": bool(
                result.selected_options is not None
                and certificate_is_feasible(instance, result.selected_options)
            ),
            "objective_value": result.objective_value,
            "upper_bound": result.objective_value if result.status == "optimal" else float("nan"),
            "lower_bound": result.objective_value if result.selected_options is not None else float("-inf"),
            "abs_gap": 0.0 if result.status == "optimal" else float("inf"),
            "rel_gap": result.gap,
            "runtime_seconds": result.runtime_seconds,
            "theta_count_total": result.theta_count_total,
            "theta_count_evaluated": result.theta_count_solved,
            "theta_count_pruned": result.theta_count_skipped,
            "robust_certificate": result.robust_certificate,
            "robust_certificate_passed": bool(
                result.selected_options is not None
                and certificate_is_feasible(instance, result.selected_options)
            ),
            "backend_available": result.available,
            "backend_message": result.message,
        }
    )
    return row


def row_key(row: Dict[str, object]) -> tuple[object, ...]:
    return (row["family"], row["n"], row["m"], row["gamma"], row["seed"], row["method"])


def load_existing(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        old = path.with_name("parametric_sweep_benchmarks.csv")
        if not old.exists():
            return []
        path = old
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows: Sequence[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with (output_dir / CSV_NAME).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    # Compatibility for previous smoke tooling.
    with (output_dir / "parametric_sweep_benchmarks.csv").open("w", newline="", encoding="utf-8") as f:
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
            theta_order = "increasing" if args.force_increasing_theta or method.startswith("exact_sweep") else "increasing"
            base = base_row(family, n, m, gamma, seed, method, theta_order)
            if row_key(base) in done:
                continue
            if method == "hullround":
                result = run_hullround(instance, base)
            elif method == "exact_enum_current":
                cfg = GlobalThetaBNBConfig(
                    time_limit_seconds=args.time_limit,
                    node_limit=args.node_limit,
                    fixed_theta_time_limit_seconds=args.fixed_theta_time_limit,
                    fixed_theta_node_limit=args.fixed_theta_node_limit,
                    collect_diagnostics=True,
                    profile_timing=True,
                    use_hullround_incumbent=True,
                    theta_order=theta_order,
                )
                result = row_from_global(base, solve_global_theta_bnb(instance, cfg))
            elif method in {"exact_sweep_new", "exact_sweep_new_validation_sampled"}:
                validate = args.validate_sweep_sampled or method == "exact_sweep_new_validation_sampled"
                cfg = GlobalThetaBNBConfig(
                    time_limit_seconds=args.time_limit,
                    node_limit=args.node_limit,
                    fixed_theta_time_limit_seconds=args.fixed_theta_time_limit,
                    fixed_theta_node_limit=args.fixed_theta_node_limit,
                    collect_diagnostics=True,
                    profile_timing=True,
                    use_hullround_incumbent=True,
                    theta_order="increasing",
                )
                sweep_cfg = ParametricThetaSweepConfig(
                    validate_against_recompute=validate,
                    max_recompute_checks=10 if validate else None,
                    reuse_hulls=True,
                    collect_diagnostics=True,
                )
                result = row_from_global(base, solve_global_theta_bnb_sweep(instance, cfg, sweep_cfg))
            elif method in {"scipy_highs", "scip", "gurobi", "cplex"}:
                result = run_optional_baseline(instance, method, base, args.solver_time_limit)
            else:
                result = dict(base)
                result.update({"status": "error", "backend_message": f"unknown method {method}"})
            rows.append(result)
            done.add(row_key(result))
            write_rows(rows, args.output_dir)
            if args.fail_fast and result["status"] == "error":
                raise RuntimeError(str(result["backend_message"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--paper-lite", action="store_true")
    parser.add_argument("--paper-full", action="store_true")
    parser.add_argument("--families", default="many_theta,tight_capacity,non_tight_control")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--time-limit", type=float, default=45.0)
    parser.add_argument("--node-limit", type=int, default=300000)
    parser.add_argument("--fixed-theta-time-limit", type=float, default=None)
    parser.add_argument("--fixed-theta-node-limit", type=int, default=None)
    parser.add_argument("--force-increasing-theta", action="store_true")
    parser.add_argument("--validate-sweep-sampled", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "parametric_sweep")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--solver-time-limit", type=float, default=5.0)
    args = parser.parse_args()
    if args.smoke:
        args.time_limit = min(args.time_limit, 5.0)
        args.node_limit = min(args.node_limit, 20000)
        args.solver_time_limit = min(args.solver_time_limit, 3.0)
    rows = run(args)
    write_rows(rows, args.output_dir)
    print(f"Wrote {args.output_dir / CSV_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
