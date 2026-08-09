#!/usr/bin/env python3
"""Run fixed-θ exact B&B benchmarks for the Path A prototype."""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT, ROOT / "src", ROOT / "experiments_nested"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments_nested._common import build_prefix_instance, make_master_portfolio  # noqa: E402
from robust_mckp import PricingInstance, solve  # noqa: E402
from robust_mckp.exact_bnb import (  # noqa: E402
    FixedThetaBNBConfig,
    brute_force_fixed_theta,
    build_fixed_theta_data,
    cost_for_selection,
    objective_for_selection,
    solve_fixed_theta_bnb,
)
from robust_mckp.greedy import greedy_lp  # noqa: E402
from robust_mckp.hull import build_upper_hull  # noqa: E402
from robust_mckp.rounding import round_lp_solution  # noqa: E402


OUT_DIR = ROOT / "results" / "exact_bnb"
CSV_PATH = OUT_DIR / "fixed_theta_bnb_benchmarks.csv"
SUMMARY_PATH = OUT_DIR / "fixed_theta_bnb_summary.md"


def raw_theta_candidates(instance: PricingInstance) -> np.ndarray:
    vals = [0.0]
    for group in instance.items:
        vals.extend(abs(opt.uncertainty) for opt in group)
    return np.unique(np.array(vals, dtype=float))


def fixed_theta_hullround(instance: PricingInstance, theta: float) -> Dict[str, object]:
    data = build_fixed_theta_data(instance, theta)
    if data.capacity < -1e-9:
        return {"status": "infeasible_capacity", "objective": float("nan")}
    hulls = []
    for i, (v, c) in enumerate(zip(data.values, data.costs)):
        hulls.append(build_upper_hull(c, v, np.arange(len(v), dtype=int)))
    try:
        lp = greedy_lp(hulls, data.capacity)
        discrete = round_lp_solution(lp, hulls, data.capacity, upgrade_completion=True)
    except Exception as exc:
        return {"status": "error", "objective": float("nan"), "error": str(exc)}
    if discrete is None:
        return {"status": "no_solution", "objective": float("nan")}
    selections = [int(hulls[i].option_indices[idx]) for i, idx in enumerate(discrete.vertices)]
    used = cost_for_selection(data.costs, selections)
    objective = objective_for_selection(data.values, selections)
    return {
        "status": "feasible" if used <= data.capacity + 1e-8 else "capacity_violation",
        "objective": objective,
        "used_capacity": used,
        "capacity_slack": data.capacity - used,
        "lp_bound": float(lp.lp_value),
        "selections": selections,
    }


def solve_fixed_theta_highs_optional(instance: PricingInstance, theta: float, time_limit: float) -> Dict[str, object]:
    try:
        import scipy.optimize as opt
    except Exception as exc:
        return {"status": "unavailable", "objective": float("nan"), "error": str(exc)}
    if not hasattr(opt, "milp"):
        return {"status": "unavailable", "objective": float("nan"), "error": "scipy.optimize.milp unavailable"}

    data = build_fixed_theta_data(instance, theta)
    if data.capacity < -1e-9:
        return {"status": "infeasible_capacity", "objective": float("nan")}
    sizes = [len(v) for v in data.values]
    total = int(sum(sizes))
    c = -np.concatenate(data.values)
    integrality = np.ones(total, dtype=int)
    bounds = opt.Bounds(lb=np.zeros(total), ub=np.ones(total))
    a_eq = np.zeros((len(sizes), total), dtype=float)
    offset = 0
    for i, m_i in enumerate(sizes):
        a_eq[i, offset : offset + m_i] = 1.0
        offset += m_i
    constraints = [
        opt.LinearConstraint(a_eq, np.ones(len(sizes)), np.ones(len(sizes))),
        opt.LinearConstraint(np.concatenate(data.costs)[None, :], -np.inf, np.array([data.capacity])),
    ]
    t0 = time.perf_counter()
    res = opt.milp(
        c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": float(time_limit), "mip_rel_gap": 1e-8},
    )
    runtime = time.perf_counter() - t0
    status_code = int(getattr(res, "status", -999))
    status_map = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded", 4: "other"}
    objective = -float(res.fun) if getattr(res, "fun", None) is not None else float("nan")
    return {
        "status": status_map.get(status_code, f"status_{status_code}"),
        "objective": objective,
        "runtime_seconds": runtime,
        "mip_gap": float(getattr(res, "mip_gap", float("nan"))),
        "message": str(getattr(res, "message", "")),
    }


def solve_fixed_theta_scip_optional(instance: PricingInstance, theta: float, time_limit: float) -> Dict[str, object]:
    try:
        from scripts.run_solver_benchmarks import solve_fixed_theta_scip
    except Exception as exc:
        return {"status": "unavailable", "objective": float("nan"), "error": str(exc)}
    res = solve_fixed_theta_scip(instance, theta, time_limit=time_limit, threads=1)
    return {
        "status": str(res.get("status", "unknown")).lower(),
        "objective": float(res.get("objective", float("nan"))),
        "runtime_seconds": float(res.get("runtime_s", float("nan"))),
        "mip_gap": float(res.get("mip_gap", float("nan"))),
        "certified": bool(res.get("certified", False)),
    }


def maybe_bruteforce(instance: PricingInstance, theta: float, max_combinations: int) -> Dict[str, object]:
    combinations = 1
    for group in instance.items:
        combinations *= len(group)
        if combinations > max_combinations:
            return {"status": "skipped_size", "objective": float("nan"), "combinations": combinations}
    res = brute_force_fixed_theta(instance, theta)
    return {
        "status": res.status,
        "objective": res.objective_value,
        "runtime_seconds": res.runtime_seconds,
        "combinations": combinations,
    }


def theta_grid(instance: PricingInstance, hullround_theta: float) -> List[float]:
    candidates = raw_theta_candidates(instance)
    vals = [0.0, float(candidates[len(candidates) // 2]), float(hullround_theta)]
    deduped: List[float] = []
    for val in vals:
        if not any(abs(val - seen) <= 1e-10 for seen in deduped):
            deduped.append(float(val))
    return deduped


def run(args: argparse.Namespace) -> List[Dict[str, object]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    ns = [10, 20] if args.smoke else [10, 20, 30, 50]
    ms = [8] if args.smoke else [8, 10]
    rows: List[Dict[str, object]] = []
    total = len(seeds) * len(ns) * len(ms)
    done = 0
    for seed in seeds:
        for m in ms:
            master = make_master_portfolio(seed=seed, n_max=max(ns), m_max=m, min_admissible_menu=min(8, m))
            for n in ns:
                done += 1
                gamma = int(math.floor(math.sqrt(n)))
                prefix = build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma)
                try:
                    hull_sol = solve(prefix.instance, upgrade_completion=True)
                    hull_theta = float(hull_sol.theta)
                    hull_global_obj = float(hull_sol.objective)
                except Exception:
                    hull_theta = 0.0
                    hull_global_obj = float("nan")
                print(f"[{done}/{total}] seed={seed} n={n} m={m} gamma={gamma}", flush=True)
                for theta in theta_grid(prefix.instance, hull_theta):
                    hr = fixed_theta_hullround(prefix.instance, theta)
                    cfg = FixedThetaBNBConfig(
                        time_limit_seconds=args.time_limit,
                        node_limit=args.node_limit,
                        tolerance=1e-9,
                    )
                    bnb = solve_fixed_theta_bnb(prefix.instance, theta, cfg)
                    brute = maybe_bruteforce(prefix.instance, theta, args.max_bruteforce_combinations)
                    highs = (
                        solve_fixed_theta_highs_optional(prefix.instance, theta, args.highs_time_limit)
                        if args.highs
                        else {"status": "not_run", "objective": float("nan")}
                    )
                    scip = (
                        solve_fixed_theta_scip_optional(prefix.instance, theta, args.scip_time_limit)
                        if args.scip
                        else {"status": "not_run", "objective": float("nan")}
                    )
                    brute_match = (
                        math.isfinite(float(brute["objective"]))
                        and abs(float(brute["objective"]) - bnb.objective_value) <= 1e-7
                    )
                    highs_match = (
                        highs.get("status") == "optimal"
                        and math.isfinite(float(highs["objective"]))
                        and abs(float(highs["objective"]) - bnb.objective_value) <= 1e-6
                    )
                    scip_match = (
                        bool(scip.get("certified", False))
                        and math.isfinite(float(scip["objective"]))
                        and abs(float(scip["objective"]) - bnb.objective_value) <= 1e-6
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "n": n,
                            "m": m,
                            "gamma": gamma,
                            "theta": theta,
                            "theta_kind": (
                                "hullround" if abs(theta - hull_theta) <= 1e-10 else "zero" if abs(theta) <= 1e-10 else "median"
                            ),
                            "hullround_global_theta": hull_theta,
                            "hullround_global_objective": hull_global_obj,
                            "fixed_theta_hullround_status": hr.get("status"),
                            "fixed_theta_hullround_objective": hr.get("objective", float("nan")),
                            "bnb_status": bnb.status,
                            "bnb_objective": bnb.objective_value,
                            "bnb_runtime_seconds": bnb.runtime_seconds,
                            "bnb_nodes_explored": bnb.nodes_explored,
                            "bnb_nodes_pruned_bound": bnb.nodes_pruned_bound,
                            "bnb_nodes_pruned_infeasible": bnb.nodes_pruned_infeasible,
                            "bnb_nodes_integral": bnb.nodes_integral,
                            "bnb_upper_bound": bnb.upper_bound,
                            "bnb_absolute_gap": bnb.absolute_gap,
                            "bnb_relative_gap": bnb.relative_gap,
                            "bnb_capacity": bnb.capacity,
                            "bnb_used_capacity": bnb.used_capacity,
                            "bnb_capacity_slack": bnb.fixed_theta_residual,
                            "bnb_valid_capacity": bnb.validation_flags.get("capacity_feasible", False),
                            "bruteforce_status": brute.get("status"),
                            "bruteforce_objective": brute.get("objective", float("nan")),
                            "bruteforce_match": brute_match,
                            "highs_status": highs.get("status"),
                            "highs_objective": highs.get("objective", float("nan")),
                            "highs_runtime_seconds": highs.get("runtime_seconds", float("nan")),
                            "highs_match": highs_match,
                            "scip_status": scip.get("status"),
                            "scip_objective": scip.get("objective", float("nan")),
                            "scip_runtime_seconds": scip.get("runtime_seconds", float("nan")),
                            "scip_match": scip_match,
                            "incumbent_initialized_by_hullround": False,
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

    optimal = [r for r in rows if r["bnb_status"] == "optimal"]
    limited = [r for r in rows if r["bnb_status"] in {"time_limit", "node_limit"}]
    highs_checked = [r for r in rows if r["highs_status"] == "optimal"]
    certified_cross_checks = [r for r in rows if r["bnb_status"] == "optimal" and r["highs_status"] == "optimal"]
    highs_matches = [r for r in certified_cross_checks if bool(r["highs_match"])]
    scip_checked = [r for r in rows if str(r.get("scip_status")) == "optimal"]
    scip_cross_checks = [r for r in rows if r["bnb_status"] == "optimal" and str(r.get("scip_status")) == "optimal"]
    scip_matches = [r for r in scip_cross_checks if bool(r.get("scip_match"))]
    runtimes = [float(r["bnb_runtime_seconds"]) for r in rows if math.isfinite(float(r["bnb_runtime_seconds"]))]
    nodes = [int(r["bnb_nodes_explored"]) for r in rows]

    lines = [
        "# Fixed-$\\theta$ B&B benchmark summary",
        "",
        f"Rows: {len(rows)}",
        f"B&B optimal: {len(optimal)}",
        f"B&B limited: {len(limited)}",
        f"HiGHS optimal comparisons: {len(highs_checked)}",
        f"HiGHS matches on B&B-optimal rows: {len(highs_matches)} / {len(certified_cross_checks)}",
        f"SCIP optimal comparisons: {len(scip_checked)}",
        f"SCIP matches on B&B-optimal rows: {len(scip_matches)} / {len(scip_cross_checks)}",
        f"Median B&B runtime seconds: {float(np.median(runtimes)) if runtimes else float('nan'):.4f}",
        f"Median B&B nodes explored: {float(np.median(nodes)) if nodes else float('nan'):.1f}",
        "",
        "Command configuration:",
        "",
        "```text",
        f"smoke={args.smoke}, seeds={args.seeds}, time_limit={args.time_limit}, node_limit={args.node_limit}, highs={args.highs}, scip={args.scip}",
        "```",
        "",
        "Below-hull integer options are not removed by the B&B search. Hulls are used only for LP upper bounds.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a shorter benchmark.")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=9901)
    parser.add_argument("--time-limit", type=float, default=2.0)
    parser.add_argument("--node-limit", type=int, default=5000)
    parser.add_argument("--highs-time-limit", type=float, default=5.0)
    parser.add_argument("--scip-time-limit", type=float, default=5.0)
    parser.add_argument("--no-highs", dest="highs", action="store_false")
    parser.add_argument("--no-scip", dest="scip", action="store_false")
    parser.add_argument("--max-bruteforce-combinations", type=int, default=200000)
    parser.set_defaults(highs=True, scip=True)
    args = parser.parse_args()

    rows = run(args)
    write_outputs(rows, args)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
