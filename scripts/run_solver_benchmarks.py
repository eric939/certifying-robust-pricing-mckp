#!/usr/bin/env python3
"""Run open/non-commercial MILP solver benchmarks for the robust MCKP paper."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT, ROOT / "src", ROOT / "experiments_nested"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments_nested._common import build_prefix_instance, make_master_portfolio  # noqa: E402
from robust_mckp import PricingInstance  # noqa: E402
from robust_mckp.certificate import certificate_is_feasible, compute_certificate  # noqa: E402

from scripts.run_publishable_experiments import (  # noqa: E402
    CSV_DIR,
    FIG_DIR,
    LOG_DIR,
    TABLE_DIR,
    detect_solver_availability,
    fixed_theta_costs,
    gamma_regime,
    hullround_metrics,
    raw_theta_candidates,
    save_environment,
    solve_fixed_theta_highs,
    solve_full_robust_highs,
    solve_theta_enum_highs,
    write_csv,
)


def ensure_dirs() -> None:
    for path in [CSV_DIR, FIG_DIR, LOG_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def finite(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def median(values: Sequence[float]) -> float:
    vals = finite(values)
    return float(statistics.median(vals)) if vals else float("nan")


def fmt(x: float, digits: int = 2) -> str:
    return "--" if not math.isfinite(float(x)) else f"{float(x):.{digits}f}"


def fmt_pct(x: float, digits: int = 2) -> str:
    return "--" if not math.isfinite(float(x)) else f"{100.0 * float(x):.{digits}f}\\%"


def objective(instance: PricingInstance, selections: Sequence[int]) -> float:
    return float(sum(instance.items[i][j].value for i, j in enumerate(selections)))


def status_clean(status: object) -> str:
    return str(status).upper().replace(" ", "_")


def _try_scip_threads(model, threads: int) -> None:
    for name in ["parallel/maxnthreads", "lp/threads"]:
        try:
            model.setIntParam(name, int(threads))
        except Exception:
            pass


def solve_full_robust_scip(
    instance: PricingInstance, *, time_limit: float = 600.0, threads: int = 1
) -> Dict[str, object]:
    try:
        from pyscipopt import Model, quicksum  # type: ignore
    except Exception as exc:
        return {"status": "UNAVAILABLE", "certified": False, "error": str(exc)}

    values = [[opt.value for opt in group] for group in instance.items]
    margins = [[opt.margin for opt in group] for group in instance.items]
    uncertainties = [[abs(opt.uncertainty) for opt in group] for group in instance.items]
    n = instance.n_items

    model = Model("full_robust_mckp")
    model.hideOutput()
    model.setRealParam("limits/time", float(time_limit))
    _try_scip_threads(model, threads)

    z = {}
    for i, group in enumerate(instance.items):
        for j, _ in enumerate(group):
            z[i, j] = model.addVar(vtype="B", name=f"z_{i}_{j}")
    theta = model.addVar(lb=0.0, vtype="C", name="theta")
    u = {i: model.addVar(lb=0.0, vtype="C", name=f"u_{i}") for i in range(n)}
    pi = {i: model.addVar(lb=0.0, vtype="C", name=f"pi_{i}") for i in range(n)}

    for i, group in enumerate(instance.items):
        model.addCons(quicksum(z[i, j] for j in range(len(group))) == 1.0)
        model.addCons(u[i] == quicksum(uncertainties[i][j] * z[i, j] for j in range(len(group))))
        model.addCons(pi[i] >= u[i] - theta)
    model.addCons(
        quicksum(margins[i][j] * z[i, j] for i, group in enumerate(instance.items) for j in range(len(group)))
        - float(instance.gamma) * theta
        - quicksum(pi[i] for i in range(n))
        >= 0.0
    )
    model.setObjective(
        quicksum(values[i][j] * z[i, j] for i, group in enumerate(instance.items) for j in range(len(group))),
        "maximize",
    )

    t0 = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - t0
    status = status_clean(model.getStatus())

    selections: Optional[List[int]] = None
    cert = float("nan")
    obj = float("nan")
    if model.getNSols() > 0:
        selections = [
            max(range(len(group)), key=lambda j: float(model.getVal(z[i, j])))
            for i, group in enumerate(instance.items)
        ]
        cert = compute_certificate(instance, selections)
        obj = objective(instance, selections)

    try:
        bound = float(model.getDualbound())
    except Exception:
        bound = float("nan")
    try:
        gap = float(model.getGap())
    except Exception:
        gap = float("nan")

    return {
        "status": status,
        "certified": status == "OPTIMAL"
        and selections is not None
        and certificate_is_feasible(instance, selections),
        "objective": obj,
        "best_bound": bound,
        "mip_gap": gap,
        "runtime_s": runtime,
        "certificate_value": cert,
        "selections": selections,
        "n_variables": model.getNVars(),
        "n_constraints": model.getNConss(),
    }


def solve_fixed_theta_scip(
    instance: PricingInstance, theta: float, *, time_limit: float = 120.0, threads: int = 1
) -> Dict[str, object]:
    try:
        from pyscipopt import Model, quicksum  # type: ignore
    except Exception as exc:
        return {"status": "UNAVAILABLE", "certified": False, "error": str(exc)}

    values, costs, _, capacity = fixed_theta_costs(instance, theta)
    if capacity < -1e-9:
        return {"status": "INFEASIBLE_CAPACITY", "certified": True}

    model = Model("fixed_theta_mckp")
    model.hideOutput()
    model.setRealParam("limits/time", float(time_limit))
    _try_scip_threads(model, threads)
    z = {}
    for i, vals in enumerate(values):
        for j in range(len(vals)):
            z[i, j] = model.addVar(vtype="B", name=f"z_{i}_{j}")
    for i, vals in enumerate(values):
        model.addCons(quicksum(z[i, j] for j in range(len(vals))) == 1.0)
    model.addCons(
        quicksum(float(costs[i][j]) * z[i, j] for i, vals in enumerate(values) for j in range(len(vals)))
        <= float(capacity)
    )
    model.setObjective(
        quicksum(float(values[i][j]) * z[i, j] for i, vals in enumerate(values) for j in range(len(vals))),
        "maximize",
    )
    t0 = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - t0
    status = status_clean(model.getStatus())

    selections: Optional[List[int]] = None
    cert = float("nan")
    obj = float("nan")
    if model.getNSols() > 0:
        selections = [max(range(len(vals)), key=lambda j: float(model.getVal(z[i, j]))) for i, vals in enumerate(values)]
        cert = compute_certificate(instance, selections)
        obj = objective(instance, selections)
    try:
        gap = float(model.getGap())
    except Exception:
        gap = float("nan")
    try:
        bound = float(model.getDualbound())
    except Exception:
        bound = float("nan")
    return {
        "status": status,
        "certified": status == "OPTIMAL"
        and selections is not None
        and certificate_is_feasible(instance, selections),
        "objective": obj,
        "best_bound": bound,
        "mip_gap": gap,
        "runtime_s": runtime,
        "certificate_value": cert,
        "selections": selections,
    }


def solve_theta_enum_scip(
    instance: PricingInstance,
    *,
    time_limit_per_theta: float = 10.0,
    threads: int = 1,
    max_thetas: int = 450,
) -> Dict[str, object]:
    candidates = raw_theta_candidates(instance)
    if candidates.size > max_thetas:
        return {
            "status": "SKIPPED_TOO_MANY_THETA",
            "certified": False,
            "theta_count": int(candidates.size),
            "runtime_s": 0.0,
        }
    best_obj = -math.inf
    best_sel: Optional[List[int]] = None
    evaluated = 0
    t0 = time.perf_counter()
    for theta in candidates:
        _, _, _, capacity = fixed_theta_costs(instance, float(theta))
        if capacity < -1e-9:
            continue
        evaluated += 1
        res = solve_fixed_theta_scip(instance, float(theta), time_limit=time_limit_per_theta, threads=threads)
        if not bool(res.get("certified")):
            return {
                "status": "NOT_CERTIFIED",
                "certified": False,
                "theta_count": int(candidates.size),
                "theta_evaluated_milp": evaluated,
                "runtime_s": time.perf_counter() - t0,
                "failed_status": res.get("status"),
            }
        obj = float(res["objective"])
        if obj > best_obj + 1e-9:
            best_obj = obj
            best_sel = list(res["selections"])  # type: ignore[arg-type]
    cert = compute_certificate(instance, best_sel) if best_sel is not None else float("nan")
    return {
        "status": "OPTIMAL" if best_sel is not None else "NO_FEASIBLE_THETA",
        "certified": best_sel is not None and certificate_is_feasible(instance, best_sel),
        "objective": float(best_obj),
        "runtime_s": time.perf_counter() - t0,
        "theta_count": int(candidates.size),
        "theta_evaluated_milp": evaluated,
        "certificate_value": float(cert),
        "selections": best_sel,
    }


def make_master(seed: int, n_max: int, m: int):
    return make_master_portfolio(seed=seed, n_max=n_max, m_max=m, min_admissible_menu=min(10, m))


def exact_design(smoke: bool) -> Tuple[List[int], List[int], int]:
    if smoke:
        return [9101], [12], 8
    return list(range(9101, 9106)), [30], 10


def scale_design(smoke: bool) -> Tuple[List[int], List[int], int]:
    if smoke:
        return [9201], [30], 8
    return list(range(9201, 9204)), [50, 100, 200, 500], 10


def choose_exact_opt(scip_enum: Dict[str, object], scip_full: Dict[str, object], highs_enum: Dict[str, object], highs_full: Dict[str, object]) -> Tuple[float, str]:
    if bool(scip_enum.get("certified")):
        return float(scip_enum["objective"]), "SCIP-ThetaEnum"
    if bool(scip_full.get("certified")):
        return float(scip_full["objective"]), "SCIP-FullRobust"
    if bool(highs_enum.get("certified")):
        return float(highs_enum["objective"]), "HiGHS-ThetaEnum"
    if bool(highs_full.get("certified")):
        return float(highs_full["objective"]), "HiGHS-FullRobust"
    return float("nan"), "none"


def run_exact(args, availability: Dict[str, object]) -> List[Dict[str, object]]:
    seeds, ns, m = exact_design(args.smoke)
    rows: List[Dict[str, object]] = []
    total = len(seeds) * len(ns) * 3
    done = 0
    for seed in seeds:
        master = make_master(seed, max(ns), m)
        for n in ns:
            gammas = sorted(set([0, int(math.floor(math.sqrt(n))), int(math.floor(0.1 * n))]))
            for gamma in gammas:
                done += 1
                print(f"[open-exact {done}/{total}] seed={seed} n={n} m={m} gamma={gamma}", flush=True)
                prefix = build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma)
                hm = hullround_metrics(prefix.instance, validate_lp=(len(rows) < 5))
                scip_full = (
                    solve_full_robust_scip(prefix.instance, time_limit=args.full_time_limit, threads=args.threads)
                    if availability.get("scip_available")
                    else {"status": "UNAVAILABLE", "certified": False}
                )
                scip_fixed = (
                    solve_fixed_theta_scip(prefix.instance, float(hm["theta"]), time_limit=args.fixed_time_limit, threads=args.threads)
                    if availability.get("scip_available")
                    else {"status": "UNAVAILABLE", "certified": False}
                )
                scip_enum = (
                    solve_theta_enum_scip(
                        prefix.instance,
                        time_limit_per_theta=args.theta_time_limit,
                        threads=args.threads,
                        max_thetas=args.max_thetas,
                    )
                    if availability.get("scip_available")
                    else {"status": "UNAVAILABLE", "certified": False}
                )
                highs_full = solve_full_robust_highs(prefix.instance, time_limit=args.full_time_limit)
                highs_fixed = solve_fixed_theta_highs(prefix.instance, float(hm["theta"]), time_limit=args.fixed_time_limit)
                highs_enum = solve_theta_enum_highs(
                    prefix.instance,
                    time_limit_per_theta=args.theta_time_limit,
                    max_thetas=args.max_thetas,
                )
                certified_obj, source = choose_exact_opt(scip_enum, scip_full, highs_enum, highs_full)
                gap = (
                    (certified_obj - float(hm["final_objective"])) / certified_obj
                    if math.isfinite(certified_obj) and abs(certified_obj) > 1e-9
                    else float("nan")
                )
                rows.append(
                    {
                        "seed": seed,
                        "n": n,
                        "m": m,
                        "alpha": 0.10,
                        "gamma": gamma,
                        "gamma_regime": gamma_regime(n, gamma),
                        "hullround_objective": float(hm["final_objective"]),
                        "hullround_runtime_s": float(hm["runtime_total_s"]),
                        "hullround_certificate": float(hm["certificate_value"]),
                        "hullround_theta": float(hm["theta"]),
                        "certified_optimum": certified_obj,
                        "certified_source": source,
                        "hullround_rel_gap_vs_certified": gap,
                        "scip_full_status": scip_full.get("status"),
                        "scip_full_certified": bool(scip_full.get("certified")),
                        "scip_full_objective": scip_full.get("objective", float("nan")),
                        "scip_full_bound": scip_full.get("best_bound", float("nan")),
                        "scip_full_mip_gap": scip_full.get("mip_gap", float("nan")),
                        "scip_full_runtime_s": scip_full.get("runtime_s", float("nan")),
                        "scip_full_certificate": scip_full.get("certificate_value", float("nan")),
                        "scip_enum_status": scip_enum.get("status"),
                        "scip_enum_certified": bool(scip_enum.get("certified")),
                        "scip_enum_objective": scip_enum.get("objective", float("nan")),
                        "scip_enum_runtime_s": scip_enum.get("runtime_s", float("nan")),
                        "scip_enum_certificate": scip_enum.get("certificate_value", float("nan")),
                        "scip_fixed_status": scip_fixed.get("status"),
                        "scip_fixed_certified": bool(scip_fixed.get("certified")),
                        "scip_fixed_objective": scip_fixed.get("objective", float("nan")),
                        "scip_fixed_runtime_s": scip_fixed.get("runtime_s", float("nan")),
                        "scip_fixed_certificate": scip_fixed.get("certificate_value", float("nan")),
                        "highs_full_status": highs_full.get("status"),
                        "highs_full_certified": bool(highs_full.get("certified")),
                        "highs_full_objective": highs_full.get("objective", float("nan")),
                        "highs_full_mip_gap": highs_full.get("mip_gap", float("nan")),
                        "highs_full_runtime_s": highs_full.get("runtime_s", float("nan")),
                        "highs_full_certificate": highs_full.get("certificate_value", float("nan")),
                        "highs_enum_status": highs_enum.get("status"),
                        "highs_enum_certified": bool(highs_enum.get("certified")),
                        "highs_enum_objective": highs_enum.get("objective", float("nan")),
                        "highs_enum_runtime_s": highs_enum.get("runtime_s", float("nan")),
                        "highs_enum_certificate": highs_enum.get("certificate_value", float("nan")),
                        "highs_fixed_status": highs_fixed.get("status"),
                        "highs_fixed_certified": bool(highs_fixed.get("certified")),
                        "highs_fixed_objective": highs_fixed.get("objective", float("nan")),
                        "highs_fixed_runtime_s": highs_fixed.get("runtime_s", float("nan")),
                        "highs_fixed_certificate": highs_fixed.get("certificate_value", float("nan")),
                    }
                )
    write_csv(CSV_DIR / "solver_benchmark_exact.csv", rows)
    write_csv(CSV_DIR / "solver_solution_validation.csv", rows)
    return rows


def run_scalability(args, availability: Dict[str, object]) -> List[Dict[str, object]]:
    seeds, ns, m = scale_design(args.smoke)
    rows: List[Dict[str, object]] = []
    total = len(seeds) * len(ns)
    done = 0
    for seed in seeds:
        master = make_master(seed, max(ns), m)
        for n in ns:
            gamma = int(math.floor(math.sqrt(n)))
            done += 1
            print(f"[open-scale {done}/{total}] seed={seed} n={n} m={m} gamma={gamma}", flush=True)
            prefix = build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma)
            hm = hullround_metrics(prefix.instance, validate_lp=False)
            run_generic = n <= args.scale_generic_nmax
            scip_full = (
                solve_full_robust_scip(prefix.instance, time_limit=args.scale_time_limit, threads=args.threads)
                if availability.get("scip_available") and run_generic
                else {"status": "NOT_RUN" if availability.get("scip_available") else "UNAVAILABLE", "certified": False}
            )
            scip_fixed = (
                solve_fixed_theta_scip(prefix.instance, float(hm["theta"]), time_limit=args.fixed_time_limit, threads=args.threads)
                if availability.get("scip_available") and run_generic
                else {"status": "NOT_RUN" if availability.get("scip_available") else "UNAVAILABLE", "certified": False}
            )
            highs_fixed = (
                solve_fixed_theta_highs(prefix.instance, float(hm["theta"]), time_limit=args.fixed_time_limit)
                if run_generic
                else {"status": "NOT_RUN", "certified": False}
            )
            rows.append(
                {
                    "seed": seed,
                    "n": n,
                    "m": m,
                    "alpha": 0.10,
                    "gamma": gamma,
                    "hullround_runtime_s": float(hm["runtime_total_s"]),
                    "hullround_objective": float(hm["final_objective"]),
                    "hullround_certificate": float(hm["certificate_value"]),
                    "hullround_theta": float(hm["theta"]),
                    "scip_full_status": scip_full.get("status"),
                    "scip_full_certified": bool(scip_full.get("certified")),
                    "scip_full_objective": scip_full.get("objective", float("nan")),
                    "scip_full_bound": scip_full.get("best_bound", float("nan")),
                    "scip_full_mip_gap": scip_full.get("mip_gap", float("nan")),
                    "scip_full_runtime_s": scip_full.get("runtime_s", float("nan")),
                    "scip_full_certificate": scip_full.get("certificate_value", float("nan")),
                    "scip_fixed_status": scip_fixed.get("status"),
                    "scip_fixed_certified": bool(scip_fixed.get("certified")),
                    "scip_fixed_objective": scip_fixed.get("objective", float("nan")),
                    "scip_fixed_runtime_s": scip_fixed.get("runtime_s", float("nan")),
                    "scip_fixed_certificate": scip_fixed.get("certificate_value", float("nan")),
                    "highs_fixed_status": highs_fixed.get("status"),
                    "highs_fixed_certified": bool(highs_fixed.get("certified")),
                    "highs_fixed_objective": highs_fixed.get("objective", float("nan")),
                    "highs_fixed_runtime_s": highs_fixed.get("runtime_s", float("nan")),
                    "highs_fixed_certificate": highs_fixed.get("certificate_value", float("nan")),
                }
            )
    write_csv(CSV_DIR / "solver_benchmark_scalability.csv", rows)
    return rows


def table_cell_cert(group: List[Dict[str, object]], solver_prefix: str) -> str:
    attempted = [r for r in group if str(r.get(f"{solver_prefix}_status")) not in {"NOT_RUN", "UNAVAILABLE"}]
    if not attempted:
        return r"\multicolumn{1}{c}{n/r}"
    cert = sum(1 for r in attempted if str(r.get(f"{solver_prefix}_certified")) == "True" or r.get(f"{solver_prefix}_certified") is True)
    return f"{cert}/{len(attempted)}"


def write_tables(exact_rows: List[Dict[str, object]], scale_rows: List[Dict[str, object]]) -> None:
    groups: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    for r in exact_rows:
        groups.setdefault((int(r["n"]), str(r["gamma_regime"])), []).append(r)
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"$n$ & $\Gamma$ & cert. & med. gap & max gap & HullRound s & SCIP full s & SCIP enum s & HiGHS enum s \\",
        r"\midrule",
    ]
    for (n, regime), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        gaps = [float(r["hullround_rel_gap_vs_certified"]) for r in group if math.isfinite(float(r["hullround_rel_gap_vs_certified"]))]
        cert = len(gaps)
        lines.append(
            f"{n} & {regime} & {cert}/{len(group)} & {fmt_pct(median(gaps), 2)} & "
            f"{fmt_pct(max(gaps) if gaps else float('nan'), 2)} & "
            f"{fmt(median([float(r['hullround_runtime_s']) for r in group]), 2)} & "
            f"{fmt(median([float(r['scip_full_runtime_s']) for r in group]), 2)} & "
            f"{fmt(median([float(r['scip_enum_runtime_s']) for r in group]), 2)} & "
            f"{fmt(median([float(r['highs_enum_runtime_s']) for r in group]), 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "solver_exact_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (TABLE_DIR / "exact_benchmark_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    scale_groups: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for r in scale_rows:
        scale_groups.setdefault((int(r["n"]), int(r["m"])), []).append(r)
    scale_lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$n$ & $m$ & HullRound s & median $Z$ & SCIP opt. & SCIP gap & SCIP s & HiGHS fixed \\",
        r"\midrule",
    ]
    for (n, m), group in sorted(scale_groups.items()):
        scip_gaps = [float(r["scip_full_mip_gap"]) for r in group if math.isfinite(float(r["scip_full_mip_gap"]))]
        scip_times = [float(r["scip_full_runtime_s"]) for r in group if math.isfinite(float(r["scip_full_runtime_s"]))]
        scale_lines.append(
            f"{n} & {m} & {fmt(median([float(r['hullround_runtime_s']) for r in group]), 2)} & "
            f"{fmt(median([float(r['hullround_certificate']) for r in group]), 1)} & "
            f"{table_cell_cert(group, 'scip_full')} & {fmt(median(scip_gaps), 3)} & "
            f"{fmt(median(scip_times), 2)} & {table_cell_cert(group, 'highs_fixed')} \\\\"
        )
    scale_lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "solver_scalability_table.tex").write_text("\n".join(scale_lines) + "\n", encoding="utf-8")
    (TABLE_DIR / "scalability_table.tex").write_text("\n".join(scale_lines) + "\n", encoding="utf-8")


def plot_solver_figures(exact_rows: List[Dict[str, object]], scale_rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    from scripts.run_publishable_experiments import apply_plot_style

    apply_plot_style()
    gaps = [100.0 * float(r["hullround_rel_gap_vs_certified"]) for r in exact_rows if math.isfinite(float(r["hullround_rel_gap_vs_certified"]))]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    if gaps:
        ax.boxplot([gaps], tick_labels=["HullRound"])
        ax.set_ylabel("Gap vs. certified optimum (%)")
    else:
        ax.text(0.5, 0.5, "No certified optima", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "solver_benchmark_gap.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.7, 2.7))
    ns = sorted({int(r["n"]) for r in scale_rows})
    hull = [median([float(r["hullround_runtime_s"]) for r in scale_rows if int(r["n"]) == n]) for n in ns]
    ax.plot(ns, hull, marker="o", label="HullRound")
    scip_ns, scip_t = [], []
    highs_ns, highs_t = [], []
    for n in ns:
        st = [float(r["scip_full_runtime_s"]) for r in scale_rows if int(r["n"]) == n and math.isfinite(float(r["scip_full_runtime_s"]))]
        ht = [float(r["highs_fixed_runtime_s"]) for r in scale_rows if int(r["n"]) == n and math.isfinite(float(r["highs_fixed_runtime_s"]))]
        if st:
            scip_ns.append(n)
            scip_t.append(median(st))
        if ht:
            highs_ns.append(n)
            highs_t.append(median(ht))
    if scip_t:
        ax.plot(scip_ns, scip_t, marker="s", linestyle="--", label="SCIP-FullRobust")
    if highs_t:
        ax.plot(highs_ns, highs_t, marker="^", linestyle=":", label="HiGHS-FixedTheta")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Seconds")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "solver_benchmark_runtime.pdf")
    plt.close(fig)


def _read_rows(name: str) -> List[Dict[str, str]]:
    path = CSV_DIR / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(finite(values))
    if not vals:
        return float("nan")
    idx = (len(vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def update_auto_numbers(availability: Dict[str, object]) -> None:
    gap_rows = _read_rows("synthetic_gap_replicates.csv")
    exact_rows = _read_rows("solver_benchmark_exact.csv")
    scale_rows = _read_rows("solver_benchmark_scalability.csv")
    cert_rows = _read_rows("synthetic_frontier_certificates.csv")
    retail_rows = _read_rows("retail_replicates.csv")
    retail_conc = _read_rows("retail_margin_concentration.csv")

    ratios = [float(r["l_rd_over_delta"]) for r in gap_rows] if gap_rows else []
    n_gaps = [float(r["n_gap_lp"]) for r in gap_rows] if gap_rows else []
    violations = [
        r for r in gap_rows
        if float(r["l_rd"]) > float(r["delta_v_max_theta"]) + 1e-7
    ]
    certified = [r for r in exact_rows if r.get("certified_source", "none") != "none"]
    certified_gaps = [
        float(r["hullround_rel_gap_vs_certified"])
        for r in certified
        if math.isfinite(float(r["hullround_rel_gap_vs_certified"]))
    ]
    scip_enum_speedups = [
        float(r["scip_enum_runtime_s"]) / float(r["hullround_runtime_s"])
        for r in exact_rows
        if math.isfinite(float(r.get("scip_enum_runtime_s", "nan")))
        and math.isfinite(float(r.get("hullround_runtime_s", "nan")))
        and float(r["hullround_runtime_s"]) > 0.0
    ]
    highs_enum_speedups = [
        float(r["highs_enum_runtime_s"]) / float(r["hullround_runtime_s"])
        for r in exact_rows
        if math.isfinite(float(r.get("highs_enum_runtime_s", "nan")))
        and math.isfinite(float(r.get("hullround_runtime_s", "nan")))
        and float(r["hullround_runtime_s"]) > 0.0
    ]
    scale_500 = [float(r["hullround_runtime_s"]) for r in scale_rows if int(r["n"]) == 500] if scale_rows else []
    retail_base = [r for r in retail_rows if r.get("attack_budget") == r.get("gamma")]
    retail_sqrt = [r for r in retail_base if int(r["gamma"]) == 17] if retail_base else []
    retail_gamma5 = [r for r in retail_base if int(r["gamma"]) == 5] if retail_base else []
    retail_gamma10 = [r for r in retail_base if int(r["gamma"]) == 10] if retail_base else []
    retail_box = [r for r in retail_base if int(r["gamma"]) == 300] if retail_base else []
    retail_gamma0 = [r for r in retail_conc if int(r["gamma"]) == 0] if retail_conc else []
    retail_gamma17 = [r for r in retail_conc if int(r["gamma"]) == 17] if retail_conc else []
    retail_gamma50 = [r for r in retail_conc if int(r["gamma"]) == 50] if retail_conc else []
    scip_scale_attempted = [r for r in scale_rows if r.get("scip_full_status") not in {"NOT_RUN", "UNAVAILABLE", None, ""}]
    scip_scale_opt = [r for r in scip_scale_attempted if r.get("scip_full_certified") == "True"]
    scip_gaps = [
        float(r["scip_full_mip_gap"])
        for r in scip_scale_attempted
        if math.isfinite(float(r.get("scip_full_mip_gap", "nan")))
    ]

    macros = {
        "PublishableGapSeeds": len({r["seed"] for r in gap_rows}) if gap_rows else 0,
        "PublishableGapNMax": max([int(r["n"]) for r in gap_rows], default=0),
        "GapRatioMedian": fmt(median(ratios), 3),
        "GapRatioPtfive": fmt(_quantile(ratios, 0.95), 3),
        "GapRatioMax": fmt(max(ratios) if ratios else float("nan"), 3),
        "GapBoundViolationCount": len(violations),
        "ScaledGapMedian": fmt(median(n_gaps), 3),
        "ExactCertifiedCount": len(certified),
        "ExactTotalCount": len(exact_rows),
        "ExactBenchmarkMedianGap": fmt_pct(median(certified_gaps), 2),
        "ExactBenchmarkMaxGap": fmt_pct(max(certified_gaps) if certified_gaps else float("nan"), 2),
        "ExactScipEnumSlowdownMedian": fmt(median(scip_enum_speedups), 1),
        "ExactHighsEnumSlowdownMedian": fmt(median(highs_enum_speedups), 1),
        "ScipStatusText": "available" if availability.get("scip_available") else "not available",
        "HighsStatusText": "available" if availability.get("highs_available") else "not available",
        "OpenSolverName": "SCIP/HiGHS" if availability.get("scip_available") else "HiGHS",
        "ScipLargeOptCount": len(scip_scale_opt),
        "ScipLargeAttemptCount": len(scip_scale_attempted),
        "ScipLargeMedianMipGap": fmt(median(scip_gaps), 3),
        "ScaleRuntimeNfivehundredMedian": fmt(median(scale_500), 2),
        "SyntheticFrontierSeedCount": len({r["seed"] for r in cert_rows}) if cert_rows else 0,
        "RetailSeedCount": len({r["seed"] for r in retail_rows}) if retail_rows else 0,
        "RetailSqrtGammaRevenueLossMedian": fmt_pct(
            1.0 - median([float(r["revenue_ratio"]) for r in retail_sqrt]), 2
        ),
        "RetailSqrtGammaViolationMedian": fmt_pct(median([float(r["violation_iid"]) for r in retail_sqrt]), 2),
        "RetailSqrtGammaResidualMedian": fmt(median([float(r["z_gamma"]) for r in retail_sqrt]), 2),
        "RetailSqrtGammaPriceChangesMedian": fmt(median([float(r["price_changes_vs_gamma0"]) for r in retail_sqrt]), 0),
        "RetailGammaFiveViolationMedian": fmt_pct(median([float(r["violation_iid"]) for r in retail_gamma5]), 2),
        "RetailGammaTenViolationMedian": fmt_pct(median([float(r["violation_iid"]) for r in retail_gamma10]), 2),
        "RetailBoxRevenueLossMedian": fmt_pct(1.0 - median([float(r["revenue_ratio"]) for r in retail_box]), 2),
        "RetailTopTenShareGammaZero": fmt_pct(median([float(r["top10_margin_share"]) for r in retail_gamma0]), 2),
        "RetailTopTenShareGammaSqrt": fmt_pct(median([float(r["top10_margin_share"]) for r in retail_gamma17]), 2),
        "RetailTopTenShareGammaFifty": fmt_pct(median([float(r["top10_margin_share"]) for r in retail_gamma50]), 2),
    }
    lines = ["% Auto-generated by scripts/run_solver_benchmarks.py"]
    for name, value in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    (ROOT / "results" / "publishable" / "auto_numbers.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--availability-only", action="store_true")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--scale-only", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--full-time-limit", type=float, default=60.0)
    parser.add_argument("--scale-time-limit", type=float, default=60.0)
    parser.add_argument("--fixed-time-limit", type=float, default=30.0)
    parser.add_argument("--theta-time-limit", type=float, default=2.0)
    parser.add_argument("--max-thetas", type=int, default=450)
    parser.add_argument("--scale-generic-nmax", type=int, default=100)
    args = parser.parse_args()

    ensure_dirs()
    availability = detect_solver_availability()
    availability["threads"] = int(args.threads)
    availability["full_robust_time_limit_s"] = float(args.full_time_limit)
    availability["fixed_theta_time_limit_s"] = float(args.fixed_time_limit)
    availability["theta_enum_time_limit_per_theta_s"] = float(args.theta_time_limit)
    availability["commercial_solvers_used"] = False
    availability["solver_policy"] = "SCIP if available; HiGHS open-source baseline; no Gurobi/CPLEX benchmark."
    design_meta = SimpleNamespace(solver_benchmark_script="scripts/run_solver_benchmarks.py")
    save_environment(availability, design_meta)  # type: ignore[arg-type]
    (LOG_DIR / "solver_availability.json").write_text(json.dumps(availability, indent=2, default=str), encoding="utf-8")
    print(json.dumps(availability, indent=2, default=str))
    if args.availability_only:
        return

    exact_rows: List[Dict[str, object]] = []
    scale_rows: List[Dict[str, object]] = []
    if not args.scale_only:
        exact_rows = run_exact(args, availability)
    elif (CSV_DIR / "solver_benchmark_exact.csv").exists():
        with (CSV_DIR / "solver_benchmark_exact.csv").open(newline="", encoding="utf-8") as fh:
            exact_rows = list(csv.DictReader(fh))

    if not args.exact_only:
        scale_rows = run_scalability(args, availability)
    elif (CSV_DIR / "solver_benchmark_scalability.csv").exists():
        with (CSV_DIR / "solver_benchmark_scalability.csv").open(newline="", encoding="utf-8") as fh:
            scale_rows = list(csv.DictReader(fh))

    if exact_rows and scale_rows:
        write_tables(exact_rows, scale_rows)
        plot_solver_figures(exact_rows, scale_rows)
    elif exact_rows:
        write_tables(exact_rows, [])
        plot_solver_figures(exact_rows, [])
    update_auto_numbers(availability)
    print("open-solver benchmarks complete")


if __name__ == "__main__":
    main()
