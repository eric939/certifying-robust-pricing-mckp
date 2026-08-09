#!/usr/bin/env python3
"""Run resumable publication-oriented benchmarks for robust MCKP solvers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT, ROOT / "src", ROOT / "experiments_nested"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments_nested._common import build_prefix_instance, make_master_portfolio  # noqa: E402
from robust_mckp import GlobalThetaBNBConfig, Option, PricingInstance, solve  # noqa: E402
from robust_mckp.certificate import certificate_is_feasible, compute_certificate  # noqa: E402
from robust_mckp.exact_bnb import solve_global_theta_bnb  # noqa: E402


DEFAULT_OUT = ROOT / "results" / "publication_benchmarks"
CSV_NAME = "publication_benchmarks.csv"
SUMMARY_NAME = "publication_benchmarks_summary.txt"
CONFIG_NAME = "publication_benchmarks_config.json"

ALL_FAMILIES = ["economic", "hull_compression", "adversarial", "tight_capacity", "many_theta", "boundary"]
DEFAULT_METHODS = ["hullround", "global_bnb_baseline", "global_bnb_cached_cutoff_ordered", "scip", "highs"]
ALL_METHODS = [
    "hullround",
    "global_bnb_baseline",
    "global_bnb_cached",
    "global_bnb_cached_cutoff",
    "global_bnb_cached_cutoff_ordered",
    "scip",
    "highs",
]

FIELDNAMES = [
    "run_key",
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
    "runtime_seconds",
    "nodes_explored",
    "theta_total",
    "theta_pruned",
    "theta_solved_optimal",
    "theta_limited",
    "global_lower_bound",
    "global_upper_bound",
    "absolute_gap",
    "relative_gap",
    "hullround_objective",
    "gap_vs_hullround",
    "scip_objective",
    "gap_vs_scip",
    "highs_objective",
    "gap_vs_highs",
    "root_lp_bound",
    "theta_order",
    "caching_enabled",
    "cutoff_enabled",
    "exact_full_theta_set_used",
    "valid_certificate",
    "instance_options",
    "notes",
]


def objective(instance: PricingInstance, selections: Sequence[int]) -> float:
    return float(sum(instance.items[i][int(j)].value for i, j in enumerate(selections)))


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(statistics.median(vals)) if vals else float("nan")


def gamma_for_mode(n: int, mode: str) -> int:
    if mode == "zero":
        return 0
    if mode == "sqrt":
        return int(math.floor(math.sqrt(n)))
    if mode == "ten_percent":
        return int(math.floor(0.1 * n))
    if mode == "full":
        return n
    raise ValueError(f"unknown gamma mode: {mode}")


def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return values or None


def grid(args: argparse.Namespace) -> Tuple[List[int], List[int], List[str]]:
    explicit_ns = _parse_int_list(args.n_values)
    explicit_ms = _parse_int_list(args.m_values)
    explicit_modes = args.gamma_mode
    if explicit_ns or explicit_ms or explicit_modes:
        ns = explicit_ns or [20, 30]
        ms = explicit_ms or [8]
        modes = explicit_modes or ["sqrt", "ten_percent"]
        return ns, ms, modes
    if args.stress:
        return [150, 200], [10], ["sqrt", "ten_percent"]
    if args.extended:
        return [20, 30, 50], [8, 10], ["zero", "sqrt", "ten_percent", "full"]
    return [20, 30], [8], ["sqrt", "ten_percent"]


def modes_for_family(family: str, base_modes: Sequence[str]) -> List[str]:
    if family == "boundary":
        return ["zero", "full"]
    return list(base_modes)


def _direct_instance(family: str, n: int, m: int, gamma: int, seed: int) -> PricingInstance:
    digest = hashlib.sha256(f"{family}|{n}|{m}|{gamma}|{seed}".encode("utf-8")).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16) % (2**32))
    items: List[List[Option]] = []
    base_margin = {
        "hull_compression": 18.0,
        "adversarial": 18.0,
        "tight_capacity": 9.0,
        "many_theta": 16.0,
        "boundary": 15.0,
    }.get(family, 16.0)
    for i in range(n):
        group: List[Option] = [Option(value=float(8.0 + rng.uniform(0, 4)), margin=base_margin, uncertainty=0.0)]
        max_cost = {"tight_capacity": 12.0, "many_theta": 10.0}.get(family, 9.0)
        costs = np.linspace(1.0, max_cost, max(1, m - 1))
        for k, cost in enumerate(costs, start=1):
            if family == "hull_compression":
                # Many points sit below a simple high endpoint chord, so LP hulls are small.
                endpoint_bonus = 18.0 if k == m - 1 else 0.0
                value = 9.0 + 1.2 * cost + endpoint_bonus + rng.normal(0, 0.15)
                uncertainty = float((k % 3) * 0.8)
            elif family == "adversarial":
                # Concave value curve keeps many points on the upper hull.
                value = 8.0 + 9.0 * math.sqrt(cost) + 0.03 * i + rng.normal(0, 0.05)
                uncertainty = float(0.5 + (k % 4) * 0.7)
            elif family == "tight_capacity":
                value = 8.0 + 3.0 * cost + rng.normal(0, 0.1)
                uncertainty = float(0.8 + 0.2 * k)
            elif family == "many_theta":
                value = 8.0 + 2.2 * cost + rng.normal(0, 0.1)
                uncertainty = float(0.05 * (1 + i * m + k))
            elif family == "boundary":
                value = 8.0 + 2.0 * cost + rng.normal(0, 0.1)
                uncertainty = float(1.0 + (k % 5) * 0.4)
            else:
                value = 8.0 + 2.0 * cost + rng.normal(0, 0.1)
                uncertainty = float(1.0 + (k % 3) * 0.5)
            margin_noise = rng.normal(0, 0.05)
            group.append(
                Option(
                    value=max(0.0, float(value)),
                    margin=float(base_margin - cost + margin_noise),
                    uncertainty=float(uncertainty),
                )
            )
        items.append(group)
    return PricingInstance(items=items, gamma=int(gamma), name=f"{family}_n{n}_m{m}_g{gamma}_s{seed}")


def build_instance(family: str, n: int, m: int, gamma: int, seed: int) -> PricingInstance:
    if family == "economic":
        master = make_master_portfolio(seed=seed, n_max=n, m_max=m, min_admissible_menu=min(8, m))
        return build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma).instance
    return _direct_instance(family, n, m, gamma, seed)


def solve_full_robust_scip_optional(instance: PricingInstance, time_limit: float) -> Dict[str, object]:
    try:
        from scripts.run_solver_benchmarks import solve_full_robust_scip
    except Exception as exc:
        return {"status": "unavailable", "objective": float("nan"), "certified": False, "runtime_seconds": 0.0, "error": str(exc)}
    res = solve_full_robust_scip(instance, time_limit=time_limit, threads=1)
    return {
        "status": str(res.get("status", "unknown")).lower(),
        "certified": bool(res.get("certified", False)),
        "objective": float(res.get("objective", float("nan"))),
        "runtime_seconds": float(res.get("runtime_s", float("nan"))),
        "gap": float(res.get("mip_gap", float("nan"))),
        "certificate": float(res.get("certificate_value", float("nan"))),
        "error": str(res.get("error", "")),
    }


def solve_full_robust_highs_optional(instance: PricingInstance, time_limit: float) -> Dict[str, object]:
    try:
        from scripts.run_publishable_experiments import solve_full_robust_highs
    except Exception as exc:
        return {"status": "unavailable", "objective": float("nan"), "certified": False, "runtime_seconds": 0.0, "error": str(exc)}
    res = solve_full_robust_highs(instance, time_limit=time_limit)
    return {
        "status": str(res.get("status", "unknown")).lower(),
        "certified": bool(res.get("certified", False)),
        "objective": float(res.get("objective", float("nan"))),
        "runtime_seconds": float(res.get("runtime_s", float("nan"))),
        "gap": float(res.get("mip_gap", float("nan"))),
        "certificate": float(res.get("certificate_value", float("nan"))),
        "error": str(res.get("error", "")),
    }


def existing_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {str(row["run_key"]) for row in csv.DictReader(f)}


def append_row(csv_path: Path, row: Dict[str, object]) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def empty_row(
    *,
    family: str,
    n: int,
    m: int,
    gamma: int,
    gamma_mode: str,
    seed: int,
    method: str,
    instance: PricingInstance,
    status: str,
    notes: str = "",
) -> Dict[str, object]:
    run_key = f"{family}|{n}|{m}|{gamma_mode}|{gamma}|{seed}|{method}"
    return {
        "run_key": run_key,
        "family": family,
        "n": n,
        "m": m,
        "gamma": gamma,
        "gamma_mode": gamma_mode,
        "seed": seed,
        "method": method,
        "status": status,
        "objective": float("nan"),
        "robust_certificate": float("nan"),
        "runtime_seconds": float("nan"),
        "nodes_explored": 0,
        "theta_total": 0,
        "theta_pruned": 0,
        "theta_solved_optimal": 0,
        "theta_limited": 0,
        "global_lower_bound": float("nan"),
        "global_upper_bound": float("nan"),
        "absolute_gap": float("nan"),
        "relative_gap": float("nan"),
        "hullround_objective": float("nan"),
        "gap_vs_hullround": float("nan"),
        "scip_objective": float("nan"),
        "gap_vs_scip": float("nan"),
        "highs_objective": float("nan"),
        "gap_vs_highs": float("nan"),
        "root_lp_bound": float("nan"),
        "theta_order": "",
        "caching_enabled": "",
        "cutoff_enabled": "",
        "exact_full_theta_set_used": "",
        "valid_certificate": False,
        "instance_options": instance.n_options,
        "notes": notes,
    }


def exact_method_config(method: str, time_limit: float, node_limit: int) -> GlobalThetaBNBConfig:
    if method == "global_bnb_baseline":
        return GlobalThetaBNBConfig(
            time_limit_seconds=time_limit,
            node_limit=node_limit,
            use_caches=False,
            use_objective_cutoff=False,
            theta_order="increasing",
        )
    if method == "global_bnb_cached":
        return GlobalThetaBNBConfig(
            time_limit_seconds=time_limit,
            node_limit=node_limit,
            use_caches=True,
            use_objective_cutoff=False,
            theta_order="increasing",
        )
    if method == "global_bnb_cached_cutoff":
        return GlobalThetaBNBConfig(
            time_limit_seconds=time_limit,
            node_limit=node_limit,
            use_caches=True,
            use_objective_cutoff=True,
            theta_order="increasing",
        )
    if method == "global_bnb_cached_cutoff_ordered":
        return GlobalThetaBNBConfig(
            time_limit_seconds=time_limit,
            node_limit=node_limit,
            use_caches=True,
            use_objective_cutoff=True,
            theta_order="lp_bound_desc",
        )
    raise ValueError(f"not an exact B&B method: {method}")


def run_hullround(instance: PricingInstance) -> Dict[str, object]:
    t0 = time.perf_counter()
    sol = solve(instance, upgrade_completion=True)
    runtime = time.perf_counter() - t0
    cert = compute_certificate(instance, sol.selections) if sol.selections else float("nan")
    return {
        "status": "feasible" if sol.is_feasible else "infeasible",
        "objective": float(sol.objective),
        "certificate": float(cert),
        "runtime_seconds": float(runtime),
        "valid_certificate": bool(sol.is_feasible and certificate_is_feasible(instance, sol.selections)),
    }


def fill_reference_gaps(row: Dict[str, object], refs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    hr_obj = float(refs.get("hullround", {}).get("objective", float("nan")))
    scip_obj = float(refs.get("scip", {}).get("objective", float("nan")))
    highs_obj = float(refs.get("highs", {}).get("objective", float("nan")))
    obj = float(row.get("objective", float("nan")))
    row["hullround_objective"] = hr_obj
    row["scip_objective"] = scip_obj
    row["highs_objective"] = highs_obj
    row["gap_vs_hullround"] = obj - hr_obj if math.isfinite(obj) and math.isfinite(hr_obj) else float("nan")
    row["gap_vs_scip"] = obj - scip_obj if math.isfinite(obj) and math.isfinite(scip_obj) else float("nan")
    row["gap_vs_highs"] = obj - highs_obj if math.isfinite(obj) and math.isfinite(highs_obj) else float("nan")
    return row


def run_method(
    *,
    instance: PricingInstance,
    family: str,
    n: int,
    m: int,
    gamma: int,
    gamma_mode: str,
    seed: int,
    method: str,
    args: argparse.Namespace,
    refs: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    row = empty_row(family=family, n=n, m=m, gamma=gamma, gamma_mode=gamma_mode, seed=seed, method=method, instance=instance, status="error")
    try:
        if method == "hullround":
            res = refs.get("hullround") or run_hullround(instance)
            refs["hullround"] = res
            row.update(
                status=res["status"],
                objective=res["objective"],
                robust_certificate=res["certificate"],
                runtime_seconds=res["runtime_seconds"],
                valid_certificate=res["valid_certificate"],
                notes="heuristic/certified feasible method; no optimality gap",
            )
        elif method == "scip":
            res = refs.get("scip") or solve_full_robust_scip_optional(instance, args.solver_time_limit)
            refs["scip"] = res
            row.update(
                status=res["status"],
                objective=res["objective"],
                robust_certificate=res.get("certificate", float("nan")),
                runtime_seconds=res["runtime_seconds"],
                absolute_gap=res.get("gap", float("nan")),
                valid_certificate=bool(res.get("certified", False)),
                notes=res.get("error", ""),
            )
        elif method == "highs":
            res = refs.get("highs") or solve_full_robust_highs_optional(instance, args.solver_time_limit)
            refs["highs"] = res
            row.update(
                status=res["status"],
                objective=res["objective"],
                robust_certificate=res.get("certificate", float("nan")),
                runtime_seconds=res["runtime_seconds"],
                absolute_gap=res.get("gap", float("nan")),
                valid_certificate=bool(res.get("certified", False)),
                notes=res.get("error", ""),
            )
        else:
            cfg = exact_method_config(method, args.time_limit, args.node_limit)
            exact = solve_global_theta_bnb(instance, cfg)
            root_bound = max((r.fixed_theta_lp_upper_bound for r in exact.per_theta_records), default=float("nan"))
            row.update(
                status=exact.status,
                objective=exact.objective_value,
                robust_certificate=exact.robust_certificate,
                runtime_seconds=exact.total_runtime_seconds,
                nodes_explored=exact.total_nodes_explored,
                theta_total=exact.theta_count_total,
                theta_pruned=exact.theta_count_pruned_by_bound,
                theta_solved_optimal=exact.theta_count_solved_optimal,
                theta_limited=exact.theta_count_limited,
                global_lower_bound=exact.lower_bound,
                global_upper_bound=exact.upper_bound,
                absolute_gap=exact.absolute_gap,
                relative_gap=exact.relative_gap,
                root_lp_bound=root_bound,
                theta_order=exact.config_metadata.get("theta_order", ""),
                caching_enabled=exact.config_metadata.get("use_caches", ""),
                cutoff_enabled=exact.config_metadata.get("use_objective_cutoff", ""),
                exact_full_theta_set_used=True,
                valid_certificate=bool(exact.validation_flags.get("robust_certificate_feasible", False)),
                notes=exact.message,
            )
    except Exception as exc:  # keep benchmark resumable
        row["status"] = "error"
        row["notes"] = f"{type(exc).__name__}: {exc}"
    return fill_reference_gaps(row, refs)


def selected_families(args: argparse.Namespace) -> List[str]:
    return args.family or ALL_FAMILIES


def selected_methods(args: argparse.Namespace) -> List[str]:
    return args.method or DEFAULT_METHODS


def write_config(out_dir: Path, args: argparse.Namespace) -> None:
    cfg = {
        "smoke": args.smoke,
        "extended": args.extended,
        "stress": args.stress,
        "families": selected_families(args),
        "methods": selected_methods(args),
        "seeds": args.seeds,
        "seed_start": args.seed_start,
        "time_limit": args.time_limit,
        "node_limit": args.node_limit,
        "solver_time_limit": args.solver_time_limit,
        "n_values": args.n_values,
        "m_values": args.m_values,
        "gamma_mode": args.gamma_mode,
        "max_rows": args.max_rows,
    }
    (out_dir / CONFIG_NAME).write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / CSV_NAME
    completed = existing_keys(csv_path) if args.resume or args.append else set()
    if csv_path.exists() and not (args.resume or args.append):
        csv_path.unlink()
        completed = set()
    write_config(out_dir, args)

    ns, ms, base_modes = grid(args)
    families = selected_families(args)
    methods = selected_methods(args)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    written = 0
    instance_counter = 0

    for family in families:
        for seed in seeds:
            for n in ns:
                for m in ms:
                    for gamma_mode in modes_for_family(family, base_modes):
                        gamma = gamma_for_mode(n, gamma_mode)
                        instance_counter += 1
                        print(f"[instance {instance_counter}] family={family} seed={seed} n={n} m={m} gamma={gamma} ({gamma_mode})", flush=True)
                        try:
                            instance = build_instance(family, n, m, gamma, seed)
                        except Exception as exc:
                            print(f"  instance error: {type(exc).__name__}: {exc}", flush=True)
                            continue
                        refs: Dict[str, Dict[str, object]] = {}
                        for method in methods:
                            run_key = f"{family}|{n}|{m}|{gamma_mode}|{gamma}|{seed}|{method}"
                            if run_key in completed:
                                print(f"  - {method}: skip existing", flush=True)
                                continue
                            if args.max_rows is not None and written >= args.max_rows:
                                write_summary(out_dir)
                                return
                            print(f"  - {method}", flush=True)
                            row = run_method(
                                instance=instance,
                                family=family,
                                n=n,
                                m=m,
                                gamma=gamma,
                                gamma_mode=gamma_mode,
                                seed=seed,
                                method=method,
                                args=args,
                                refs=refs,
                            )
                            append_row(csv_path, row)
                            completed.add(run_key)
                            written += 1
    write_summary(out_dir)


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_summary(out_dir: Path) -> None:
    csv_path = out_dir / CSV_NAME
    rows = read_rows(csv_path)
    by_method: Dict[str, List[Dict[str, str]]] = {}
    by_family: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
        by_family.setdefault(row["family"], []).append(row)
    lines = [
        "Publication benchmark summary",
        "",
        f"Rows: {len(rows)}",
        f"Instances: {len(set(r['run_key'].rsplit('|', 1)[0] for r in rows))}",
        f"Families: {', '.join(sorted(by_family))}",
        "",
        "By method:",
    ]
    baseline: Dict[str, float] = {}
    for row in rows:
        if row["method"] == "global_bnb_baseline":
            try:
                baseline[row["run_key"].replace("|global_bnb_baseline", "")] = float(row["runtime_seconds"])
            except Exception:
                pass
    for method, rs in sorted(by_method.items()):
        optimal = sum(1 for r in rs if r["status"] in {"optimal", "OPTIMAL", "optimal "})
        certified = sum(1 for r in rs if str(r["valid_certificate"]).lower() == "true")
        limited = sum(1 for r in rs if r["status"] in {"time_limit", "node_limit", "timelimit", "TIME_LIMIT", "userinterrupt"})
        errors = sum(1 for r in rs if r["status"] == "error")
        speedups = []
        for r in rs:
            key = r["run_key"].rsplit("|", 1)[0]
            base = baseline.get(key)
            try:
                rt = float(r["runtime_seconds"])
            except Exception:
                rt = float("nan")
            if base and base > 0 and math.isfinite(rt) and rt > 0:
                speedups.append(base / rt)
        theta_rates = []
        for r in rs:
            try:
                total = float(r["theta_total"])
                pruned = float(r["theta_pruned"])
                if total > 0:
                    theta_rates.append(pruned / total)
            except Exception:
                pass
        lines.append(
            (
                f"- {method}: rows={len(rs)}, optimal={optimal}, valid_cert={certified}, limited={limited}, "
                f"errors={errors}, median_runtime={median(float(r['runtime_seconds']) for r in rs):.4f}s, "
                f"median_nodes={median(float(r['nodes_explored']) for r in rs):.1f}, "
                f"median_theta_prune_rate={median(theta_rates):.3f}, "
                f"median_speedup_vs_baseline={median(speedups):.3f}"
            )
        )
    lines += ["", "By family:"]
    for family, rs in sorted(by_family.items()):
        lines.append(
            f"- {family}: rows={len(rs)}, methods={len(set(r['method'] for r in rs))}, median_runtime={median(float(r['runtime_seconds']) for r in rs):.4f}s"
        )
    (out_dir / SUMMARY_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    grid_group = parser.add_mutually_exclusive_group()
    grid_group.add_argument("--smoke", action="store_true")
    grid_group.add_argument("--extended", action="store_true")
    grid_group.add_argument("--stress", action="store_true")
    parser.add_argument("--family", action="append", choices=ALL_FAMILIES)
    parser.add_argument("--method", action="append", choices=ALL_METHODS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=9911)
    parser.add_argument("--n-values", default=None, help="Comma-separated n grid, e.g. 50,80,100")
    parser.add_argument("--m-values", default=None, help="Comma-separated m grid, e.g. 8,10")
    parser.add_argument(
        "--gamma-mode",
        action="append",
        choices=["zero", "sqrt", "ten_percent", "full"],
        help="Gamma regime to include; repeat for multiple modes.",
    )
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--node-limit", type=int, default=50000)
    parser.add_argument("--solver-time-limit", type=float, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    if not (args.smoke or args.extended or args.stress):
        args.smoke = True
    if args.solver_time_limit is None:
        args.solver_time_limit = args.time_limit
    run(args)


if __name__ == "__main__":
    main()
