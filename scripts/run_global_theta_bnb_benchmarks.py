#!/usr/bin/env python3
"""Run global θ-enumeration exact B&B benchmarks for Path A."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

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
CSV_PATH = OUT_DIR / "global_theta_bnb_benchmarks.csv"
SUMMARY_PATH = OUT_DIR / "global_theta_bnb_summary.md"


def objective(instance: PricingInstance, selections: Sequence[int]) -> float:
    return float(sum(instance.items[i][int(j)].value for i, j in enumerate(selections)))


def solve_full_robust_scip_optional(instance: PricingInstance, time_limit: float) -> Dict[str, object]:
    try:
        from scripts.run_solver_benchmarks import solve_full_robust_scip
    except Exception as exc:
        return {"status": "unavailable", "objective": float("nan"), "error": str(exc)}
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
        return {"status": "unavailable", "objective": float("nan"), "error": str(exc)}
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


def gamma_values(n: int) -> List[int]:
    return sorted(set([0, int(math.floor(math.sqrt(n))), int(math.floor(0.1 * n)), n]))


def run(args: argparse.Namespace) -> List[Dict[str, object]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    ns = [10] if args.smoke else [10, 20, 30]
    ms = [8] if args.smoke else [8, 10]
    rows: List[Dict[str, object]] = []
    total = sum(len(gamma_values(n)) for _seed in seeds for _m in ms for n in ns)
    done = 0

    for seed in seeds:
        for m in ms:
            master = make_master_portfolio(seed=seed, n_max=max(ns), m_max=m, min_admissible_menu=min(8, m))
            for n in ns:
                for gamma in gamma_values(n):
                    done += 1
                    prefix = build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma)
                    print(f"[{done}/{total}] seed={seed} n={n} m={m} gamma={gamma}", flush=True)
                    try:
                        hr = solve(prefix.instance, upgrade_completion=True)
                        hr_status = "feasible" if hr.is_feasible else "infeasible"
                        hr_obj = float(hr.objective)
                        hr_cert = float(hr.certificate_value)
                    except Exception as exc:
                        hr_status = f"error:{type(exc).__name__}"
                        hr_obj = float("nan")
                        hr_cert = float("nan")

                    cfg = GlobalThetaBNBConfig(
                        time_limit_seconds=args.time_limit,
                        node_limit=args.node_limit,
                        fixed_theta_time_limit_seconds=args.fixed_theta_time_limit,
                        fixed_theta_node_limit=args.fixed_theta_node_limit,
                        use_hullround_incumbent=True,
                    )
                    exact = solve_global_theta_bnb(prefix.instance, cfg)
                    brute = maybe_bruteforce(prefix.instance, args.max_bruteforce_combinations)
                    scip = (
                        solve_full_robust_scip_optional(prefix.instance, args.solver_time_limit)
                        if args.scip
                        else {"status": "not_run", "objective": float("nan"), "certified": False}
                    )
                    highs = (
                        solve_full_robust_highs_optional(prefix.instance, args.solver_time_limit)
                        if args.highs
                        else {"status": "not_run", "objective": float("nan"), "certified": False}
                    )

                    brute_match = (
                        math.isfinite(float(brute.get("objective", float("nan"))))
                        and abs(float(brute["objective"]) - exact.objective_value) <= 1e-7
                    )
                    scip_match = (
                        bool(scip.get("certified", False))
                        and exact.status == "optimal"
                        and abs(float(scip["objective"]) - exact.objective_value) <= 1e-6
                    )
                    highs_match = (
                        bool(highs.get("certified", False))
                        and exact.status == "optimal"
                        and abs(float(highs["objective"]) - exact.objective_value) <= 1e-6
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "n": n,
                            "m": m,
                            "gamma": gamma,
                            "hullround_status": hr_status,
                            "hullround_objective": hr_obj,
                            "hullround_certificate": hr_cert,
                            "global_bnb_status": exact.status,
                            "global_bnb_objective": exact.objective_value,
                            "global_bnb_certificate": exact.robust_certificate,
                            "global_bnb_runtime_seconds": exact.total_runtime_seconds,
                            "global_bnb_lower_bound": exact.lower_bound,
                            "global_bnb_upper_bound": exact.upper_bound,
                            "global_bnb_absolute_gap": exact.absolute_gap,
                            "global_bnb_relative_gap": exact.relative_gap,
                            "theta_total": exact.theta_count_total,
                            "theta_pruned": exact.theta_count_pruned_by_bound,
                            "theta_infeasible_capacity": exact.theta_count_infeasible_capacity,
                            "theta_solved_optimal": exact.theta_count_solved_optimal,
                            "theta_limited": exact.theta_count_limited,
                            "theta_error": exact.theta_count_error,
                            "nodes_total": exact.total_nodes_explored,
                            "improvement_over_hullround": (
                                exact.objective_value - hr_obj if math.isfinite(hr_obj) and math.isfinite(exact.objective_value) else float("nan")
                            ),
                            "certified_optimum": exact.status == "optimal",
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
    return rows


def write_outputs(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> None:
    if not rows:
        return
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    statuses = {}
    for r in rows:
        statuses[str(r["global_bnb_status"])] = statuses.get(str(r["global_bnb_status"]), 0) + 1
    certified = [r for r in rows if r["global_bnb_status"] == "optimal"]
    runtimes = [float(r["global_bnb_runtime_seconds"]) for r in rows if math.isfinite(float(r["global_bnb_runtime_seconds"]))]
    nodes = [int(r["nodes_total"]) for r in rows]
    gaps = [float(r["global_bnb_absolute_gap"]) for r in rows if math.isfinite(float(r["global_bnb_absolute_gap"]))]
    scip_checked = [r for r in rows if bool(r["scip_certified"]) and r["global_bnb_status"] == "optimal"]
    highs_checked = [r for r in rows if bool(r["highs_certified"]) and r["global_bnb_status"] == "optimal"]

    lines = [
        "# Global $\\theta$-B&B benchmark summary",
        "",
        f"Rows: {len(rows)}",
        f"Status counts: {statuses}",
        f"Certified global optima: {len(certified)}",
        f"SCIP matches on certified rows: {sum(1 for r in scip_checked if bool(r['scip_match']))} / {len(scip_checked)}",
        f"HiGHS matches on certified rows: {sum(1 for r in highs_checked if bool(r['highs_match']))} / {len(highs_checked)}",
        f"Median runtime seconds: {float(np.median(runtimes)) if runtimes else float('nan'):.4f}",
        f"Median nodes explored: {float(np.median(nodes)) if nodes else float('nan'):.1f}",
        f"Max reported absolute gap: {max(gaps) if gaps else float('nan'):.6g}",
        "",
        "Command configuration:",
        "",
        "```text",
        (
            f"smoke={args.smoke}, seeds={args.seeds}, time_limit={args.time_limit}, "
            f"node_limit={args.node_limit}, fixed_theta_time_limit={args.fixed_theta_time_limit}, "
            f"fixed_theta_node_limit={args.fixed_theta_node_limit}, highs={args.highs}, scip={args.scip}"
        ),
        "```",
        "",
        "The solver uses the full theta candidate set `{0} union {|t_ij|}`.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=9911)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--node-limit", type=int, default=50000)
    parser.add_argument("--fixed-theta-time-limit", type=float, default=None)
    parser.add_argument("--fixed-theta-node-limit", type=int, default=None)
    parser.add_argument("--solver-time-limit", type=float, default=5.0)
    parser.add_argument("--max-bruteforce-combinations", type=int, default=200000)
    parser.add_argument("--no-highs", dest="highs", action="store_false")
    parser.add_argument("--no-scip", dest="scip", action="store_false")
    parser.set_defaults(highs=True, scip=True)
    args = parser.parse_args()

    rows = run(args)
    write_outputs(rows, args)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
