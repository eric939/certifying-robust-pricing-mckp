#!/usr/bin/env python3
"""Run targeted tight-capacity diagnostics for exact global θ enumeration."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT, ROOT / "src", ROOT / "experiments_nested"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robust_mckp import GlobalThetaBNBConfig, PricingInstance, solve_global_theta_bnb  # noqa: E402
from robust_mckp.certificate import compute_certificate  # noqa: E402
from scripts.run_publication_benchmarks import (  # noqa: E402
    build_instance,
    gamma_for_mode,
    run_hullround,
    solve_full_robust_highs_optional,
    solve_full_robust_scip_optional,
)


DEFAULT_OUT = ROOT / "results" / "tight_capacity_diagnostics"
CSV_NAME = "tight_capacity_diagnostics.csv"
JSON_NAME = "tight_capacity_diagnostics.json"
SUMMARY_NAME = "tight_capacity_diagnostics_summary.txt"

FIELDNAMES = [
    "family",
    "n",
    "m",
    "gamma",
    "gamma_mode",
    "seed",
    "method",
    "status",
    "objective",
    "robust_certificate",
    "lower_bound",
    "upper_bound",
    "absolute_gap",
    "relative_gap",
    "runtime_seconds",
    "theta_total",
    "theta_pruned_by_root_lp",
    "theta_solved_optimal",
    "theta_limited",
    "theta_error",
    "total_nodes_explored",
    "total_nodes_pruned_bound",
    "total_nodes_pruned_infeasible",
    "total_nodes_pruned_cutoff",
    "total_integral_lp_nodes",
    "total_fractional_branches",
    "total_fallback_branches",
    "average_branching_arity",
    "max_branching_arity",
    "time_root_lp_total",
    "time_node_lp_total",
    "time_hull_build_total",
    "time_greedy_lp_total",
    "time_fast_bound_total",
    "time_reference_bound_total",
    "time_cheap_prebound_total",
    "time_min_cost_check_total",
    "time_branching_total",
    "time_child_generation_total",
    "cheap_prebound_prunes",
    "min_cost_infeasibility_prunes",
    "fast_lp_bounds_computed",
    "exact_lp_bounds_computed",
    "reference_fallback_count",
    "bound_cache_hits",
    "bound_cache_misses",
    "hardest_theta",
    "hardest_theta_gap",
    "hardest_theta_nodes",
    "hardest_theta_runtime",
    "unresolved_theta_count",
    "best_remaining_theta_upper_bound",
    "notes",
]


def _parse_int_list(raw: Optional[str], default: Sequence[int]) -> List[int]:
    if raw is None:
        return list(default)
    vals = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return vals or list(default)


def _finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _median(values: Iterable[float]) -> float:
    vals = _finite(values)
    return float(statistics.median(vals)) if vals else float("nan")


def _status_is_limited(status: str) -> bool:
    return status in {"time_limit", "node_limit", "not_run_time_limit", "not_run_node_limit"}


def _row_key(row: Dict[str, object]) -> str:
    return "|".join(str(row[k]) for k in ["family", "n", "m", "gamma_mode", "gamma", "seed", "method"])


def _existing_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {_row_key(row) for row in csv.DictReader(f)}


def _append_row(csv_path: Path, row: Dict[str, object]) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def _load_json(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_json(path: Path, records: List[Dict[str, object]]) -> None:
    path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")


def _hardest_theta(records) -> Dict[str, object]:
    if not records:
        return {
            "theta": float("nan"),
            "gap": float("nan"),
            "nodes": 0,
            "runtime": float("nan"),
            "unresolved_count": 0,
            "best_upper": float("nan"),
        }
    unresolved = [
        r
        for r in records
        if _status_is_limited(r.status)
        or (math.isfinite(float(r.bnb_upper_bound)) and math.isfinite(float(r.bnb_lower_bound)) and r.bnb_upper_bound > r.bnb_lower_bound + 1e-9)
    ]
    pool = unresolved or list(records)
    hardest = max(
        pool,
        key=lambda r: (
            float("-inf") if not math.isfinite(float(r.bnb_gap)) else float(r.bnb_gap),
            int(r.nodes_explored),
            float(r.runtime_seconds),
            -float(r.theta),
        ),
    )
    best_upper = max((float(r.bnb_upper_bound) for r in unresolved if math.isfinite(float(r.bnb_upper_bound))), default=float("nan"))
    return {
        "theta": float(hardest.theta),
        "gap": float(hardest.bnb_gap),
        "nodes": int(hardest.nodes_explored),
        "runtime": float(hardest.runtime_seconds),
        "unresolved_count": len(unresolved),
        "best_upper": best_upper,
    }


def _objective(instance: PricingInstance, selections: Optional[Sequence[int]]) -> float:
    if selections is None:
        return float("nan")
    return float(sum(instance.items[i][int(j)].value for i, j in enumerate(selections)))


def _exact_row(
    *,
    instance: PricingInstance,
    n: int,
    m: int,
    gamma: int,
    gamma_mode: str,
    seed: int,
    method: str,
    args: argparse.Namespace,
) -> tuple[Dict[str, object], Dict[str, object]]:
    use_fast = method != "global_bnb_cached_cutoff_ordered_reference_bound"
    cfg = GlobalThetaBNBConfig(
        time_limit_seconds=float(args.time_limit),
        node_limit=int(args.node_limit),
        use_caches=True,
        use_objective_cutoff=True,
        theta_order="lp_bound_desc",
        collect_diagnostics=True,
        profile_timing=True,
        max_diagnostic_nodes=int(args.max_diagnostic_nodes),
        diagnostic_sample_rate=int(args.diagnostic_sample_rate),
        use_fast_residual_lp_bound=use_fast,
        use_cheap_prebound=use_fast,
        use_min_cost_infeasibility_check=True,
        use_bound_cache=bool(args.use_bound_cache and use_fast),
    )
    res = solve_global_theta_bnb(instance, cfg)
    diag = res.diagnostics
    hard = _hardest_theta(res.per_theta_records)
    row = {
        "family": "tight_capacity",
        "n": n,
        "m": m,
        "gamma": gamma,
        "gamma_mode": gamma_mode,
        "seed": seed,
        "method": method,
        "status": res.status,
        "objective": res.objective_value,
        "robust_certificate": res.robust_certificate,
        "lower_bound": res.lower_bound,
        "upper_bound": res.upper_bound,
        "absolute_gap": res.absolute_gap,
        "relative_gap": res.relative_gap,
        "runtime_seconds": res.total_runtime_seconds,
        "theta_total": res.theta_count_total,
        "theta_pruned_by_root_lp": res.theta_count_pruned_by_bound,
        "theta_solved_optimal": res.theta_count_solved_optimal,
        "theta_limited": res.theta_count_limited,
        "theta_error": res.theta_count_error,
        "total_nodes_explored": res.total_nodes_explored,
        "total_nodes_pruned_bound": diag.get("total_nodes_pruned_bound", 0),
        "total_nodes_pruned_infeasible": diag.get("total_nodes_pruned_infeasible", 0),
        "total_nodes_pruned_cutoff": diag.get("total_nodes_pruned_cutoff", 0),
        "total_integral_lp_nodes": diag.get("total_integral_lp_nodes", 0),
        "total_fractional_branches": diag.get("total_fractional_branches", 0),
        "total_fallback_branches": diag.get("total_fallback_branches", 0),
        "average_branching_arity": diag.get("average_branching_arity", float("nan")),
        "max_branching_arity": diag.get("max_branching_arity", 0),
        "time_root_lp_total": res.total_root_lp_time_seconds,
        "time_node_lp_total": diag.get("total_time_node_lp", 0.0),
        "time_hull_build_total": diag.get("total_time_hull_build", 0.0),
        "time_greedy_lp_total": diag.get("total_time_greedy_lp", 0.0),
        "time_fast_bound_total": diag.get("total_time_fast_bound", 0.0),
        "time_reference_bound_total": diag.get("total_time_reference_bound", 0.0),
        "time_cheap_prebound_total": diag.get("total_time_cheap_prebound", 0.0),
        "time_min_cost_check_total": diag.get("total_time_min_cost_check", 0.0),
        "time_branching_total": diag.get("total_time_branching", 0.0),
        "time_child_generation_total": diag.get("total_time_child_generation", 0.0),
        "cheap_prebound_prunes": diag.get("total_cheap_prebound_prunes", 0),
        "min_cost_infeasibility_prunes": diag.get("total_min_cost_infeasibility_prunes", 0),
        "fast_lp_bounds_computed": diag.get("total_fast_lp_bounds_computed", 0),
        "exact_lp_bounds_computed": diag.get("total_exact_lp_bounds_computed", 0),
        "reference_fallback_count": diag.get("total_reference_fallback_count", 0),
        "bound_cache_hits": diag.get("total_bound_cache_hits", 0),
        "bound_cache_misses": diag.get("total_bound_cache_misses", 0),
        "hardest_theta": hard["theta"],
        "hardest_theta_gap": hard["gap"],
        "hardest_theta_nodes": hard["nodes"],
        "hardest_theta_runtime": hard["runtime"],
        "unresolved_theta_count": hard["unresolved_count"],
        "best_remaining_theta_upper_bound": hard["best_upper"],
        "notes": res.message,
    }
    detail = {
        "key": _row_key(row),
        "config": res.config_metadata,
        "result": {
            "status": res.status,
            "objective": res.objective_value,
            "lower_bound": res.lower_bound,
            "upper_bound": res.upper_bound,
            "relative_gap": res.relative_gap,
            "robust_certificate": res.robust_certificate,
        },
        "diagnostics": res.diagnostics,
        "per_theta": [
            {
                "theta": r.theta,
                "status": r.status,
                "capacity": r.fixed_theta_capacity,
                "root_lp_upper_bound": r.fixed_theta_lp_upper_bound,
                "incumbent_before": r.incumbent_before_theta,
                "incumbent_after": r.incumbent_after_theta,
                "lower_bound": r.bnb_lower_bound,
                "upper_bound": r.bnb_upper_bound,
                "gap": r.bnb_gap,
                "nodes": r.nodes_explored,
                "runtime_seconds": r.runtime_seconds,
                "diagnostics": r.diagnostics,
            }
            for r in res.per_theta_records
        ],
    }
    return row, detail


def _simple_method_row(
    *,
    method: str,
    instance: PricingInstance,
    n: int,
    m: int,
    gamma: int,
    gamma_mode: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Dict[str, object], Optional[Dict[str, object]]]:
    row = {field: "" for field in FIELDNAMES}
    row.update(
        family="tight_capacity",
        n=n,
        m=m,
        gamma=gamma,
        gamma_mode=gamma_mode,
        seed=seed,
        method=method,
        status="error",
        notes="",
    )
    try:
        if method == "hullround":
            res = run_hullround(instance)
            row.update(
                status=res["status"],
                objective=res["objective"],
                robust_certificate=res["certificate"],
                runtime_seconds=res["runtime_seconds"],
                notes="heuristic robust-feasible incumbent; no optimality certificate",
            )
        elif method == "scip":
            res = solve_full_robust_scip_optional(instance, args.solver_time_limit)
            row.update(
                status=res["status"],
                objective=res["objective"],
                robust_certificate=res.get("certificate", float("nan")),
                runtime_seconds=res["runtime_seconds"],
                absolute_gap=res.get("gap", float("nan")),
                notes=res.get("error", ""),
            )
        elif method == "highs":
            res = solve_full_robust_highs_optional(instance, args.solver_time_limit)
            row.update(
                status=res["status"],
                objective=res["objective"],
                robust_certificate=res.get("certificate", float("nan")),
                runtime_seconds=res["runtime_seconds"],
                absolute_gap=res.get("gap", float("nan")),
                notes=res.get("error", ""),
            )
    except Exception as exc:
        row["status"] = "error"
        row["notes"] = f"{type(exc).__name__}: {exc}"
    return row, None


def _write_summary(out_dir: Path) -> None:
    csv_path = out_dir / CSV_NAME
    rows: List[Dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    exact_rows = [r for r in rows if r.get("method", "").startswith("global_bnb_cached_cutoff_ordered")]
    statuses: Dict[str, int] = {}
    for row in exact_rows:
        statuses[row.get("status", "")] = statuses.get(row.get("status", ""), 0) + 1
    runtimes = _finite(float(r.get("runtime_seconds", "nan")) for r in exact_rows)
    gaps = _finite(float(r.get("relative_gap", "nan")) for r in exact_rows)
    node_lp_times = _finite(float(r.get("time_node_lp_total", "nan")) for r in exact_rows)
    lines = [
        "Tight-capacity diagnostics summary",
        f"Rows: {len(rows)}",
        f"Exact diagnostic rows: {len(exact_rows)}",
        f"Exact statuses: {statuses}",
        f"Median exact runtime (s): {_median(runtimes):.6g}",
        f"Median exact relative gap: {_median(gaps):.6g}",
        f"Median node LP time (s): {_median(node_lp_times):.6g}",
    ]
    by_method: Dict[str, List[Dict[str, str]]] = {}
    for row in exact_rows:
        by_method.setdefault(row.get("method", ""), []).append(row)
    if by_method:
        lines.append("Exact rows by method:")
        for method, method_rows in sorted(by_method.items()):
            method_statuses: Dict[str, int] = {}
            for row in method_rows:
                method_statuses[row.get("status", "")] = method_statuses.get(row.get("status", ""), 0) + 1
            lines.append(
                f"  {method}: rows={len(method_rows)} statuses={method_statuses} "
                f"median_runtime={_median(float(r.get('runtime_seconds', 'nan')) for r in method_rows):.6g} "
                f"median_node_lp={_median(float(r.get('time_node_lp_total', 'nan')) for r in method_rows):.6g} "
                f"median_gap={_median(float(r.get('relative_gap', 'nan')) for r in method_rows):.6g}"
            )
    limited = [r for r in exact_rows if _status_is_limited(r.get("status", ""))]
    if limited:
        lines.append("Limited exact rows:")
        for r in limited:
            lines.append(
                "  "
                + f"n={r['n']} m={r['m']} gamma_mode={r['gamma_mode']} seed={r['seed']} "
                + f"gap={r.get('relative_gap')} nodes={r.get('total_nodes_explored')} hardest_theta={r.get('hardest_theta')}"
            )
    (out_dir / SUMMARY_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.smoke:
        if float(args.time_limit) == 45.0:
            args.time_limit = 20.0
        if int(args.node_limit) == 300000:
            args.node_limit = 100000
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / CSV_NAME
    json_path = out_dir / JSON_NAME
    if csv_path.exists() and not args.resume:
        csv_path.unlink()
    completed = _existing_keys(csv_path) if args.resume else set()
    details = _load_json(json_path) if args.resume else []
    detail_keys = {str(d.get("key", "")) for d in details}

    ns = [50] if args.smoke else _parse_int_list(args.n_values, [50, 80, 100])
    ms = [8] if args.smoke else _parse_int_list(args.m_values, [8, 10])
    gamma_modes = ["sqrt"] if args.smoke else (args.gamma_mode or ["sqrt", "ten_percent"])
    seeds = [args.seed_start] if args.smoke else list(range(args.seed_start, args.seed_start + args.seeds))
    methods = args.method or ["global_bnb_cached_cutoff_ordered_fast_bound", "hullround"]

    written = 0
    for seed in seeds:
        for n in ns:
            for m in ms:
                for gamma_mode in gamma_modes:
                    gamma = gamma_for_mode(n, gamma_mode)
                    print(f"[instance] tight_capacity seed={seed} n={n} m={m} gamma={gamma} ({gamma_mode})", flush=True)
                    instance = build_instance("tight_capacity", n, m, gamma, seed)
                    for method in methods:
                        key = f"tight_capacity|{n}|{m}|{gamma_mode}|{gamma}|{seed}|{method}"
                        if key in completed:
                            print(f"  - {method}: skip existing", flush=True)
                            continue
                        if args.max_rows is not None and written >= args.max_rows:
                            _write_json(json_path, details)
                            _write_summary(out_dir)
                            return
                        print(f"  - {method}", flush=True)
                        if method.startswith("global_bnb_cached_cutoff_ordered"):
                            row, detail = _exact_row(instance=instance, n=n, m=m, gamma=gamma, gamma_mode=gamma_mode, seed=seed, method=method, args=args)
                        else:
                            row, detail = _simple_method_row(
                                method=method,
                                instance=instance,
                                n=n,
                                m=m,
                                gamma=gamma,
                                gamma_mode=gamma_mode,
                                seed=seed,
                                args=args,
                            )
                        _append_row(csv_path, row)
                        if detail is not None and str(detail.get("key", "")) not in detail_keys:
                            details.append(detail)
                            detail_keys.add(str(detail.get("key", "")))
                            _write_json(json_path, details)
                        written += 1
    _write_json(json_path, details)
    _write_summary(out_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run one small diagnostic instance.")
    parser.add_argument("--n-values", default=None, help="Comma-separated n values.")
    parser.add_argument("--m-values", default=None, help="Comma-separated m values.")
    parser.add_argument("--gamma-mode", action="append", choices=["sqrt", "ten_percent", "zero", "full"], help="Gamma mode; repeatable.")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=45.0)
    parser.add_argument("--node-limit", type=int, default=300000)
    parser.add_argument("--solver-time-limit", type=float, default=45.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--method",
        action="append",
        choices=[
            "global_bnb_cached_cutoff_ordered",
            "global_bnb_cached_cutoff_ordered_reference_bound",
            "global_bnb_cached_cutoff_ordered_fast_bound",
            "hullround",
            "scip",
            "highs",
        ],
        help="Method to run; repeatable.",
    )
    parser.add_argument("--methods", dest="method", action="append", help=argparse.SUPPRESS)
    parser.add_argument("--use-bound-cache", action="store_true", help="Enable the optional fixed-node bound cache for fast-bound exact rows.")
    parser.add_argument("--collect-node-samples", action="store_true", help="Compatibility flag; node samples are collected for exact rows.")
    parser.add_argument("--max-diagnostic-nodes", type=int, default=20000)
    parser.add_argument("--diagnostic-sample-rate", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
