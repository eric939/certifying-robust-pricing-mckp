#!/usr/bin/env python3
"""Run performance benchmarks for exact θ-enumeration B&B configurations."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT, ROOT / "src", ROOT / "experiments_nested"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments_nested._common import build_prefix_instance, make_master_portfolio  # noqa: E402
from robust_mckp import GlobalThetaBNBConfig, PricingInstance, solve  # noqa: E402
from robust_mckp.certificate import compute_certificate  # noqa: E402
from robust_mckp.exact_bnb import brute_force_global_robust, solve_global_theta_bnb  # noqa: E402


OUT_DIR = ROOT / "results" / "exact_bnb"
CSV_PATH = OUT_DIR / "performance_benchmarks.csv"
SUMMARY_PATH = OUT_DIR / "performance_benchmarks_summary.md"
DIAGNOSTICS_PATH = OUT_DIR / "performance_diagnostics.json"


def objective(instance: PricingInstance, selections: Sequence[int]) -> float:
    return float(sum(instance.items[i][int(j)].value for i, j in enumerate(selections)))


def gamma_values(n: int) -> List[int]:
    return sorted(set([0, int(math.floor(math.sqrt(n))), int(math.floor(0.1 * n)), n]))


def grid_from_args(args: argparse.Namespace) -> Tuple[List[int], List[int]]:
    if args.stress:
        return [150, 200], [10]
    if args.extended:
        return [30, 50, 80, 100], [8, 10, 20]
    return [10, 20, 30], [8, 10]


def method_configs(args: argparse.Namespace) -> List[Tuple[str, Dict[str, object]]]:
    ordered_name = f"cached_cutoff_{args.theta_order}"
    return [
        (
            "baseline_uncached_increasing",
            {"use_caches": False, "use_objective_cutoff": False, "theta_order": "increasing"},
        ),
        (
            "cached_increasing",
            {"use_caches": True, "use_objective_cutoff": False, "theta_order": "increasing"},
        ),
        (
            "cached_cutoff_increasing",
            {"use_caches": True, "use_objective_cutoff": True, "theta_order": "increasing"},
        ),
        (
            ordered_name,
            {"use_caches": True, "use_objective_cutoff": True, "theta_order": args.theta_order},
        ),
    ]


def solve_full_robust_scip_optional(instance: PricingInstance, time_limit: float) -> Dict[str, object]:
    try:
        from scripts.run_solver_benchmarks import solve_full_robust_scip
    except Exception as exc:
        return {"status": "unavailable", "objective": float("nan"), "certified": False, "error": str(exc)}
    res = solve_full_robust_scip(instance, time_limit=time_limit, threads=1)
    return {
        "status": str(res.get("status", "unknown")).lower(),
        "certified": bool(res.get("certified", False)),
        "objective": float(res.get("objective", float("nan"))),
        "runtime_seconds": float(res.get("runtime_s", float("nan"))),
        "gap": float(res.get("mip_gap", float("nan"))),
    }


def solve_full_robust_highs_optional(instance: PricingInstance, time_limit: float) -> Dict[str, object]:
    try:
        from scripts.run_publishable_experiments import solve_full_robust_highs
    except Exception as exc:
        return {"status": "unavailable", "objective": float("nan"), "certified": False, "error": str(exc)}
    res = solve_full_robust_highs(instance, time_limit=time_limit)
    return {
        "status": str(res.get("status", "unknown")).lower(),
        "certified": bool(res.get("certified", False)),
        "objective": float(res.get("objective", float("nan"))),
        "runtime_seconds": float(res.get("runtime_s", float("nan"))),
        "gap": float(res.get("mip_gap", float("nan"))),
    }


def maybe_bruteforce(instance: PricingInstance, max_combinations: int) -> Dict[str, object]:
    combinations = 1
    for group in instance.items:
        combinations *= len(group)
        if combinations > max_combinations:
            return {"status": "skipped_size", "objective": float("nan"), "combinations": combinations}
    brute = brute_force_global_robust(instance)
    return {
        "status": brute.status,
        "objective": brute.objective_value,
        "runtime_seconds": brute.total_runtime_seconds,
        "certificate": brute.robust_certificate,
        "combinations": combinations,
    }


def median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(vals)) if vals else float("nan")


def run(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ns, ms = grid_from_args(args)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    methods = method_configs(args)
    rows: List[Dict[str, object]] = []
    diagnostics_rows: List[Dict[str, object]] = []
    total_instances = sum(len(gamma_values(n)) for _seed in seeds for _m in ms for n in ns)
    done = 0

    for seed in seeds:
        for m in ms:
            master = make_master_portfolio(seed=seed, n_max=max(ns), m_max=m, min_admissible_menu=min(8, m))
            for n in ns:
                for gamma in gamma_values(n):
                    done += 1
                    prefix = build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma)
                    instance = prefix.instance
                    instance_key = f"seed={seed}|n={n}|m={m}|gamma={gamma}"
                    print(f"[{done}/{total_instances}] {instance_key}", flush=True)

                    try:
                        hr = solve(instance, upgrade_completion=True)
                        hr_status = "feasible" if hr.is_feasible else "infeasible"
                        hr_obj = float(hr.objective)
                        hr_cert = float(hr.certificate_value)
                        hr_runtime = float(hr.elapsed)
                    except Exception as exc:
                        hr_status = f"error:{type(exc).__name__}"
                        hr_obj = float("nan")
                        hr_cert = float("nan")
                        hr_runtime = float("nan")

                    brute = maybe_bruteforce(instance, args.max_bruteforce_combinations)
                    scip = (
                        solve_full_robust_scip_optional(instance, args.solver_time_limit)
                        if args.scip
                        else {"status": "not_run", "objective": float("nan"), "certified": False}
                    )
                    highs = (
                        solve_full_robust_highs_optional(instance, args.solver_time_limit)
                        if args.highs
                        else {"status": "not_run", "objective": float("nan"), "certified": False}
                    )

                    for method, overrides in methods:
                        print(f"  - {method}", flush=True)
                        cfg = GlobalThetaBNBConfig(
                            time_limit_seconds=args.time_limit,
                            node_limit=args.node_limit,
                            use_hullround_incumbent=True,
                            use_fixed_theta_greedy_incumbent=True,
                            collect_diagnostics=args.collect_diagnostics,
                            **overrides,
                        )
                        t0 = time.perf_counter()
                        exact = solve_global_theta_bnb(instance, cfg)
                        wall = time.perf_counter() - t0
                        brute_match = (
                            math.isfinite(float(brute.get("objective", float("nan"))))
                            and exact.status == "optimal"
                            and abs(float(brute["objective"]) - exact.objective_value) <= 1e-7
                        )
                        scip_match = (
                            bool(scip.get("certified", False))
                            and exact.status == "optimal"
                            and abs(float(scip.get("objective", float("nan"))) - exact.objective_value) <= 1e-6
                        )
                        highs_match = (
                            bool(highs.get("certified", False))
                            and exact.status == "optimal"
                            and abs(float(highs.get("objective", float("nan"))) - exact.objective_value) <= 1e-6
                        )
                        theta_pruning_rate = (
                            exact.theta_count_pruned_by_bound / exact.theta_count_total if exact.theta_count_total else float("nan")
                        )
                        root_gap = (
                            exact.upper_bound - exact.lower_bound if math.isfinite(exact.upper_bound) and math.isfinite(exact.lower_bound) else float("nan")
                        )
                        rows.append(
                            {
                                "instance_key": instance_key,
                                "method": method,
                                "seed": seed,
                                "n": n,
                                "m": m,
                                "gamma": gamma,
                                "theta_order": exact.config_metadata.get("theta_order"),
                                "use_caches": exact.config_metadata.get("use_caches"),
                                "use_objective_cutoff": exact.config_metadata.get("use_objective_cutoff"),
                                "status": exact.status,
                                "objective": exact.objective_value,
                                "robust_certificate": exact.robust_certificate,
                                "runtime_seconds": exact.total_runtime_seconds,
                                "wall_runtime_seconds": wall,
                                "root_lp_time_seconds": exact.total_root_lp_time_seconds,
                                "fixed_theta_bnb_time_seconds": exact.total_fixed_theta_bnb_time_seconds,
                                "lower_bound": exact.lower_bound,
                                "upper_bound": exact.upper_bound,
                                "absolute_gap": exact.absolute_gap,
                                "relative_gap": exact.relative_gap,
                                "theta_total": exact.theta_count_total,
                                "theta_infeasible_capacity": exact.theta_count_infeasible_capacity,
                                "theta_pruned": exact.theta_count_pruned_by_bound,
                                "theta_solved_optimal": exact.theta_count_solved_optimal,
                                "theta_limited": exact.theta_count_limited,
                                "theta_error": exact.theta_count_error,
                                "theta_pruning_rate": theta_pruning_rate,
                                "nodes_total": exact.total_nodes_explored,
                                "root_lp_bound_gap": root_gap,
                                "hullround_status": hr_status,
                                "hullround_objective": hr_obj,
                                "hullround_certificate": hr_cert,
                                "hullround_runtime_seconds": hr_runtime,
                                "improvement_over_hullround": (
                                    exact.objective_value - hr_obj
                                    if math.isfinite(exact.objective_value) and math.isfinite(hr_obj)
                                    else float("nan")
                                ),
                                "bruteforce_status": brute.get("status"),
                                "bruteforce_objective": brute.get("objective", float("nan")),
                                "bruteforce_match": brute_match,
                                "scip_status": scip.get("status"),
                                "scip_certified": bool(scip.get("certified", False)),
                                "scip_objective": scip.get("objective", float("nan")),
                                "scip_runtime_seconds": scip.get("runtime_seconds", float("nan")),
                                "scip_match": scip_match,
                                "highs_status": highs.get("status"),
                                "highs_certified": bool(highs.get("certified", False)),
                                "highs_objective": highs.get("objective", float("nan")),
                                "highs_runtime_seconds": highs.get("runtime_seconds", float("nan")),
                                "highs_match": highs_match,
                            }
                        )
                        if args.collect_diagnostics:
                            diagnostics_rows.append(
                                {
                                    "instance_key": instance_key,
                                    "method": method,
                                    "global_diagnostics": exact.diagnostics,
                                    "per_theta": [r.__dict__ for r in exact.per_theta_records[: args.max_diagnostic_theta]],
                                }
                            )
    return rows, diagnostics_rows


def write_outputs(rows: Sequence[Dict[str, object]], diagnostics_rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> None:
    if not rows:
        return
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    baseline_by_instance = {
        str(r["instance_key"]): float(r["runtime_seconds"])
        for r in rows
        if r["method"] == "baseline_uncached_increasing" and math.isfinite(float(r["runtime_seconds"]))
    }
    lines = [
        "# Exact B&B performance benchmark summary",
        "",
        f"Rows: {len(rows)}",
        f"Instances: {len(set(str(r['instance_key']) for r in rows))}",
        "",
        "## By Method",
        "",
        "| Method | Rows | Optimal | Limited | Errors | Median runtime (s) | Median nodes | Median theta prune rate | Median final gap | Median speedup vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, method_rows in by_method.items():
        optimal = sum(1 for r in method_rows if r["status"] == "optimal")
        limited = sum(1 for r in method_rows if r["status"] in {"time_limit", "node_limit"})
        errors = sum(1 for r in method_rows if r["status"] == "error")
        speedups = []
        for r in method_rows:
            base = baseline_by_instance.get(str(r["instance_key"]))
            runtime = float(r["runtime_seconds"])
            if base and base > 0 and math.isfinite(runtime) and runtime > 0:
                speedups.append(base / runtime)
        lines.append(
            "| {method} | {rows} | {optimal} | {limited} | {errors} | {runtime:.4f} | {nodes:.1f} | {prune:.3f} | {gap:.6g} | {speedup:.3f} |".format(
                method=method,
                rows=len(method_rows),
                optimal=optimal,
                limited=limited,
                errors=errors,
                runtime=median(float(r["runtime_seconds"]) for r in method_rows),
                nodes=median(float(r["nodes_total"]) for r in method_rows),
                prune=median(float(r["theta_pruning_rate"]) for r in method_rows),
                gap=median(float(r["absolute_gap"]) for r in method_rows),
                speedup=median(speedups),
            )
        )

    lines += [
        "",
        "## Solver Checks",
        "",
    ]
    for method, method_rows in by_method.items():
        certified = [r for r in method_rows if r["status"] == "optimal"]
        scip_checked = [r for r in certified if bool(r["scip_certified"])]
        highs_checked = [r for r in certified if bool(r["highs_certified"])]
        brute_checked = [r for r in certified if math.isfinite(float(r["bruteforce_objective"]))]
        lines += [
            f"- `{method}`: certified {len(certified)} / {len(method_rows)} rows.",
            f"- `{method}`: brute-force matches {sum(1 for r in brute_checked if bool(r['bruteforce_match']))} / {len(brute_checked)} checked rows.",
            f"- `{method}`: SCIP matches {sum(1 for r in scip_checked if bool(r['scip_match']))} / {len(scip_checked)} certified rows with SCIP certificates.",
            f"- `{method}`: HiGHS matches {sum(1 for r in highs_checked if bool(r['highs_match']))} / {len(highs_checked)} certified rows with HiGHS certificates.",
        ]

    lines += [
        "",
        "## Command Configuration",
        "",
        "```text",
        (
            f"smoke={args.smoke}, extended={args.extended}, stress={args.stress}, seeds={args.seeds}, "
            f"time_limit={args.time_limit}, node_limit={args.node_limit}, theta_order={args.theta_order}, "
            f"highs={args.highs}, scip={args.scip}, collect_diagnostics={args.collect_diagnostics}"
        ),
        "```",
        "",
        "All exact configurations use the full theta candidate set `{0} union {|t_ij|}`.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if diagnostics_rows:
        DIAGNOSTICS_PATH.write_text(json.dumps(diagnostics_rows, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument("--smoke", action="store_true")
    grid.add_argument("--extended", action="store_true")
    grid.add_argument("--stress", action="store_true")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=9911)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--node-limit", type=int, default=50000)
    parser.add_argument("--theta-order", choices=["lp_bound_desc", "hybrid", "heuristic_incumbent_desc"], default="lp_bound_desc")
    parser.add_argument("--solver-time-limit", type=float, default=5.0)
    parser.add_argument("--max-bruteforce-combinations", type=int, default=200000)
    parser.add_argument("--collect-diagnostics", action="store_true")
    parser.add_argument("--max-diagnostic-theta", type=int, default=200)
    parser.add_argument("--no-scip", dest="scip", action="store_false")
    parser.add_argument("--no-highs", dest="highs", action="store_false")
    parser.set_defaults(scip=True, highs=True)
    args = parser.parse_args()
    if not (args.smoke or args.extended or args.stress):
        args.smoke = True

    rows, diagnostics_rows = run(args)
    write_outputs(rows, diagnostics_rows, args)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    if diagnostics_rows:
        print(f"Wrote {DIAGNOSTICS_PATH}")


if __name__ == "__main__":
    main()
