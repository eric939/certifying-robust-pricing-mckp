#!/usr/bin/env python3
"""Compare exact-safe branching rules on tight-capacity robust MCKP rows."""
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
from scripts.run_publication_benchmarks import build_instance, gamma_for_mode  # noqa: E402


DEFAULT_OUT = ROOT / "results" / "branching_diagnostics"
CSV_NAME = "branching_diagnostics.csv"
JSON_NAME = "branching_diagnostics.json"
SUMMARY_NAME = "branching_diagnostics_summary.txt"

DEFAULT_BRANCHING_RULES = [
    "fractional_item_then_spread",
    "largest_hull_jump",
    "largest_cost_spread",
    "tight_capacity_hybrid",
    "strong_branching_lite",
]

FIELDNAMES = [
    "family",
    "n",
    "m",
    "gamma",
    "gamma_mode",
    "seed",
    "branch_rule",
    "child_ordering",
    "incumbent_heuristic",
    "status",
    "objective",
    "lower_bound",
    "upper_bound",
    "absolute_gap",
    "relative_gap",
    "runtime_seconds",
    "nodes_explored",
    "nodes_pruned_bound",
    "nodes_pruned_infeasible",
    "nodes_pruned_cutoff",
    "strong_branching_count",
    "strong_branching_time",
    "strong_branching_candidates_evaluated",
    "average_branching_arity",
    "node_lp_time",
    "fast_bound_time",
    "branching_time",
    "child_generation_time",
    "fractional_branches",
    "fallback_branches",
    "local_incumbent_swaps",
    "local_incumbent_improved",
    "theta_total",
    "theta_pruned",
    "theta_solved_optimal",
    "theta_limited",
    "final_robust_certificate",
    "valid_certificate",
    "notes",
]


def _parse_int_list(raw: Optional[str], default: Sequence[int]) -> List[int]:
    if raw is None:
        return list(default)
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return values or list(default)


def _finite(values: Iterable[object]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            val = float(value)
        except Exception:
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def _median(values: Iterable[object]) -> float:
    vals = _finite(values)
    return float(statistics.median(vals)) if vals else float("nan")


def _limited(status: str) -> bool:
    return status in {"time_limit", "node_limit", "not_run_time_limit", "not_run_node_limit"}


def _row_key(row: Dict[str, object]) -> str:
    return "|".join(str(row[k]) for k in ["family", "n", "m", "gamma_mode", "gamma", "seed", "branch_rule"])


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {_row_key(row) for row in csv.DictReader(f)}


def _append_row(path: Path, row: Dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
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


def _diag(res, key: str, default: object = 0) -> object:
    return res.diagnostics.get(key, default)


def _run_rule(
    *,
    instance: PricingInstance,
    n: int,
    m: int,
    gamma: int,
    gamma_mode: str,
    seed: int,
    branch_rule: str,
    args: argparse.Namespace,
) -> tuple[Dict[str, object], Dict[str, object]]:
    cfg = GlobalThetaBNBConfig(
        time_limit_seconds=float(args.time_limit),
        node_limit=int(args.node_limit),
        use_caches=True,
        use_objective_cutoff=True,
        theta_order="lp_bound_desc",
        use_fast_residual_lp_bound=True,
        use_cheap_prebound=True,
        use_min_cost_infeasibility_check=True,
        branching_rule=branch_rule,
        child_ordering=str(args.child_ordering),
        strong_branching_candidates=int(args.strong_branching_candidates),
        strong_branching_depth_limit=int(args.strong_branching_depth_limit),
        strong_branching_max_children=int(args.strong_branching_max_children),
        incumbent_heuristic=str(args.incumbent_heuristic),
        use_local_incumbent_improvement=not bool(args.no_local_improvement),
        local_search_max_passes=int(args.local_search_max_passes),
        local_search_max_swaps=int(args.local_search_max_swaps),
        collect_diagnostics=True,
        profile_timing=True,
        max_diagnostic_nodes=int(args.max_diagnostic_nodes),
        diagnostic_sample_rate=int(args.diagnostic_sample_rate),
    )
    res = solve_global_theta_bnb(instance, cfg)
    row = {
        "family": "tight_capacity",
        "n": n,
        "m": m,
        "gamma": gamma,
        "gamma_mode": gamma_mode,
        "seed": seed,
        "branch_rule": branch_rule,
        "child_ordering": args.child_ordering,
        "incumbent_heuristic": args.incumbent_heuristic,
        "status": res.status,
        "objective": res.objective_value,
        "lower_bound": res.lower_bound,
        "upper_bound": res.upper_bound,
        "absolute_gap": res.absolute_gap,
        "relative_gap": res.relative_gap,
        "runtime_seconds": res.total_runtime_seconds,
        "nodes_explored": res.total_nodes_explored,
        "nodes_pruned_bound": _diag(res, "total_nodes_pruned_bound", 0),
        "nodes_pruned_infeasible": _diag(res, "total_nodes_pruned_infeasible", 0),
        "nodes_pruned_cutoff": _diag(res, "total_nodes_pruned_cutoff", 0),
        "strong_branching_count": _diag(res, "total_strong_branching_count", 0),
        "strong_branching_time": _diag(res, "total_strong_branching_time", 0.0),
        "strong_branching_candidates_evaluated": _diag(res, "total_strong_branching_candidates_evaluated", 0),
        "average_branching_arity": _diag(res, "average_branching_arity", float("nan")),
        "node_lp_time": _diag(res, "total_time_node_lp", 0.0),
        "fast_bound_time": _diag(res, "total_time_fast_bound", 0.0),
        "branching_time": _diag(res, "total_time_branching", 0.0),
        "child_generation_time": _diag(res, "total_time_child_generation", 0.0),
        "fractional_branches": _diag(res, "total_fractional_branches", 0),
        "fallback_branches": _diag(res, "total_fallback_branches", 0),
        "local_incumbent_swaps": _diag(res, "total_local_incumbent_swaps", 0),
        "local_incumbent_improved": bool(_diag(res, "total_local_incumbent_swaps", 0)),
        "theta_total": res.theta_count_total,
        "theta_pruned": res.theta_count_pruned_by_bound,
        "theta_solved_optimal": res.theta_count_solved_optimal,
        "theta_limited": res.theta_count_limited,
        "final_robust_certificate": res.robust_certificate,
        "valid_certificate": bool(res.validation_flags.get("robust_certificate_feasible", False)),
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
            "certificate": res.robust_certificate,
        },
        "diagnostics": res.diagnostics,
        "per_theta": [
            {
                "theta": rec.theta,
                "status": rec.status,
                "lower_bound": rec.bnb_lower_bound,
                "upper_bound": rec.bnb_upper_bound,
                "gap": rec.bnb_gap,
                "nodes": rec.nodes_explored,
                "runtime_seconds": rec.runtime_seconds,
                "diagnostics": rec.diagnostics,
            }
            for rec in res.per_theta_records
        ],
    }
    return row, detail


def _write_summary(out_dir: Path) -> None:
    path = out_dir / CSV_NAME
    rows: List[Dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    lines = ["Branching diagnostics summary", f"Rows: {len(rows)}"]
    by_rule: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_rule.setdefault(row.get("branch_rule", ""), []).append(row)
    for rule, rule_rows in sorted(by_rule.items()):
        statuses: Dict[str, int] = {}
        for row in rule_rows:
            statuses[row.get("status", "")] = statuses.get(row.get("status", ""), 0) + 1
        lines.append(
            f"{rule}: rows={len(rule_rows)} statuses={statuses} "
            f"median_gap={_median(r.get('relative_gap', 'nan') for r in rule_rows):.6g} "
            f"median_runtime={_median(r.get('runtime_seconds', 'nan') for r in rule_rows):.6g} "
            f"median_nodes={_median(r.get('nodes_explored', 'nan') for r in rule_rows):.6g} "
            f"median_strong_time={_median(r.get('strong_branching_time', 'nan') for r in rule_rows):.6g}"
        )
    limited = [r for r in rows if _limited(r.get("status", ""))]
    if limited:
        lines.append("Limited rows:")
        for row in limited:
            lines.append(
                "  "
                + f"n={row.get('n')} m={row.get('m')} gamma_mode={row.get('gamma_mode')} "
                + f"seed={row.get('seed')} rule={row.get('branch_rule')} "
                + f"gap={row.get('relative_gap')} nodes={row.get('nodes_explored')}"
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
    rules = args.branching_rule or DEFAULT_BRANCHING_RULES

    written = 0
    for seed in seeds:
        for n in ns:
            for m in ms:
                for gamma_mode in gamma_modes:
                    gamma = gamma_for_mode(n, gamma_mode)
                    print(f"[instance] tight_capacity seed={seed} n={n} m={m} gamma={gamma} ({gamma_mode})", flush=True)
                    instance = build_instance("tight_capacity", n, m, gamma, seed)
                    for rule in rules:
                        key = f"tight_capacity|{n}|{m}|{gamma_mode}|{gamma}|{seed}|{rule}"
                        if key in completed:
                            print(f"  - {rule}: skip existing", flush=True)
                            continue
                        if args.max_rows is not None and written >= args.max_rows:
                            _write_json(json_path, details)
                            _write_summary(out_dir)
                            return
                        print(f"  - {rule}", flush=True)
                        try:
                            row, detail = _run_rule(
                                instance=instance,
                                n=n,
                                m=m,
                                gamma=gamma,
                                gamma_mode=gamma_mode,
                                seed=seed,
                                branch_rule=rule,
                                args=args,
                            )
                        except Exception as exc:
                            row = {field: "" for field in FIELDNAMES}
                            row.update(
                                family="tight_capacity",
                                n=n,
                                m=m,
                                gamma=gamma,
                                gamma_mode=gamma_mode,
                                seed=seed,
                                branch_rule=rule,
                                child_ordering=args.child_ordering,
                                incumbent_heuristic=args.incumbent_heuristic,
                                status="error",
                                notes=f"{type(exc).__name__}: {exc}",
                            )
                            detail = None
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
    parser.add_argument("--gamma-mode", action="append", choices=["sqrt", "ten_percent", "zero", "full"])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=45.0)
    parser.add_argument("--node-limit", type=int, default=300000)
    parser.add_argument("--branching-rule", action="append", choices=DEFAULT_BRANCHING_RULES)
    parser.add_argument("--child-ordering", choices=["value_desc", "cost_asc", "density_desc", "bound_promise"], default="value_desc")
    parser.add_argument("--incumbent-heuristic", default="greedy")
    parser.add_argument("--no-local-improvement", action="store_true")
    parser.add_argument("--local-search-max-passes", type=int, default=2)
    parser.add_argument("--local-search-max-swaps", type=int, default=1000)
    parser.add_argument("--strong-branching-candidates", type=int, default=5)
    parser.add_argument("--strong-branching-depth-limit", type=int, default=3)
    parser.add_argument("--strong-branching-max-children", type=int, default=50)
    parser.add_argument("--max-diagnostic-nodes", type=int, default=20000)
    parser.add_argument("--diagnostic-sample-rate", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
