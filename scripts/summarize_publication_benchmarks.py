#!/usr/bin/env python3
"""Generate TeX summary tables for publication benchmark CSV outputs."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METHOD_LABELS = {
    "hullround": "HullRound",
    "global_bnb_baseline": "Global $\\theta$-B\\&B",
    "global_bnb_cached": "Global $\\theta$-B\\&B + cache",
    "global_bnb_cached_cutoff": "Global $\\theta$-B\\&B + cache/cutoff",
    "global_bnb_cached_cutoff_ordered": "Global $\\theta$-B\\&B + cache/cutoff/order",
    "scip": "SCIP robust MILP",
    "highs": "HiGHS robust MILP",
}

METHOD_ORDER = [
    "hullround",
    "global_bnb_baseline",
    "global_bnb_cached_cutoff_ordered",
    "scip",
    "highs",
]

FAMILY_LABELS = {
    "economic": "Economic",
    "hull_compression": "Hull compression",
    "adversarial": "Low compression",
    "tight_capacity": "Tight capacity",
    "many_theta": "Many $\\theta$",
    "boundary": "Boundary $\\Gamma$",
}

FAMILY_ORDER = ["economic", "hull_compression", "adversarial", "tight_capacity", "many_theta", "boundary"]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def value(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def finite(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            out.append(fv)
    return out


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(statistics.median(vals)) if vals else float("nan")


def fmt_num(x: float, digits: int = 4) -> str:
    if not math.isfinite(x):
        return "--"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{digits}f}"


def fmt_int(x: float) -> str:
    if not math.isfinite(x):
        return "--"
    return str(int(round(x)))


def is_limited(status: str) -> bool:
    return status in {"time_limit", "node_limit", "timelimit", "userinterrupt"}


def write_table(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def method_summary(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    lines = [
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Method & Rows & Cert./feas. & Limited & Median time (s) & Median nodes & Median gap \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        rs = [r for r in rows if r["method"] == method]
        if not rs:
            continue
        if method == "hullround":
            certified = sum(1 for r in rs if r["status"] == "feasible" and r.get("valid_certificate", "").lower() == "true")
        else:
            certified = sum(1 for r in rs if r["status"] == "optimal")
        limited = sum(1 for r in rs if is_limited(r["status"]))
        med_time = median(value(r, "runtime_seconds") for r in rs)
        med_nodes = median(value(r, "nodes_explored") for r in rs if method.startswith("global_bnb"))
        med_gap = median(value(r, "absolute_gap") for r in rs if math.isfinite(value(r, "absolute_gap")))
        lines.append(
            f"{METHOD_LABELS.get(method, method)} & {len(rs)} & {certified} & {limited} & "
            f"{fmt_num(med_time)} & {fmt_int(med_nodes)} & {fmt_num(med_gap)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_table(out_dir / "exact_solver_publication_summary.tex", lines)


def run_key_without_method(row: Dict[str, str]) -> Tuple[str, str, str, str, str, str]:
    return (row["family"], row["n"], row["m"], row["gamma"], row["gamma_mode"], row["seed"])


def family_summary(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    optimized = [r for r in rows if r["method"] == "global_bnb_cached_cutoff_ordered"]
    hull = {run_key_without_method(r): r for r in rows if r["method"] == "hullround"}
    lines = [
        "\\begin{tabular}{@{}lrrrrrll@{}}",
        "\\toprule",
        "Family & Inst. & Cert. & Limited & Med. time (s) & Med. $\\theta$ pruned & HullRound gap & Note \\\\",
        "\\midrule",
    ]
    for family in FAMILY_ORDER:
        rs = [r for r in optimized if r["family"] == family]
        if not rs:
            continue
        inst = len(rs)
        cert = sum(1 for r in rs if r["status"] == "optimal")
        limited = sum(1 for r in rs if is_limited(r["status"]))
        med_time = median(value(r, "runtime_seconds") for r in rs)
        prune_rates = []
        gaps = []
        for r in rs:
            total = value(r, "theta_total")
            pruned = value(r, "theta_pruned")
            if total > 0:
                prune_rates.append(pruned / total)
            if r["status"] == "optimal":
                h = hull.get(run_key_without_method(r))
                obj = value(r, "objective")
                hr = value(h, "objective") if h else float("nan")
                if obj > 0 and math.isfinite(hr):
                    gaps.append(max(0.0, (obj - hr) / obj))
        med_prune = median(prune_rates)
        med_gap = median(gaps)
        gap_text = f"{fmt_num(100.0 * med_gap, 4)}\\%" if math.isfinite(med_gap) else "--"
        note = "tight rows limit" if family == "tight_capacity" and limited else "--"
        lines.append(
            f"{FAMILY_LABELS.get(family, family)} & {inst} & {cert} & {limited} & "
            f"{fmt_num(med_time)} & {fmt_num(100.0 * med_prune, 1)}\\% & {gap_text} & {note} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_table(out_dir / "family_diagnostics_summary.tex", lines)


def status_summary(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    statuses = ["optimal", "feasible", "time_limit", "node_limit", "timelimit", "userinterrupt", "error", "infeasible"]
    lines = [
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        "Method & Optimal & Feasible & Time & Node & Error & Infeas. & Total \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        rs = [r for r in rows if r["method"] == method]
        if not rs:
            continue
        c = Counter(r["status"] for r in rs)
        time_limited = c["time_limit"] + c["timelimit"] + c["userinterrupt"]
        lines.append(
            f"{METHOD_LABELS.get(method, method)} & {c['optimal']} & {c['feasible']} & {time_limited} & "
            f"{c['node_limit']} & {c['error']} & {c['infeasible']} & {len(rs)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_table(out_dir / "method_status_summary.tex", lines)


def method_note(method: str, certified: int, limited: int) -> str:
    if method == "hullround":
        return "heuristic"
    if limited:
        return "valid gaps"
    if certified:
        return "certified"
    return "--"


def staged_solver_summary(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    lines = [
        "\\begin{tabular}{@{}lrrrrrrl@{}}",
        "\\toprule",
        "Method & Rows & Cert./feas. & Limited & Median time (s) & Median nodes & Median rel. gap & Notes \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        rs = [r for r in rows if r["method"] == method]
        if not rs:
            continue
        if method == "hullround":
            certified = sum(1 for r in rs if r["status"] == "feasible" and r.get("valid_certificate", "").lower() == "true")
        else:
            certified = sum(1 for r in rs if r["status"] == "optimal")
        limited = sum(1 for r in rs if is_limited(r["status"]))
        med_time = median(value(r, "runtime_seconds") for r in rs)
        med_nodes = median(value(r, "nodes_explored") for r in rs if method.startswith("global_bnb"))
        med_rel_gap = median(value(r, "relative_gap") for r in rs if math.isfinite(value(r, "relative_gap")))
        lines.append(
            f"{METHOD_LABELS.get(method, method)} & {len(rs)} & {certified} & {limited} & "
            f"{fmt_num(med_time)} & {fmt_int(med_nodes)} & {fmt_num(med_rel_gap)} & "
            f"{method_note(method, certified, limited)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_table(out_dir / "staged_solver_summary.tex", lines)


def hullround_gap_for_family(rows: Sequence[Dict[str, str]], family: str) -> float:
    optimized = [r for r in rows if r["method"] == "global_bnb_cached_cutoff_ordered" and r["family"] == family and r["status"] == "optimal"]
    hull = {run_key_without_method(r): r for r in rows if r["method"] == "hullround"}
    gaps = []
    for r in optimized:
        h = hull.get(run_key_without_method(r))
        obj = value(r, "objective")
        hr = value(h, "objective") if h else float("nan")
        if obj > 0 and math.isfinite(hr):
            gaps.append(max(0.0, (obj - hr) / obj))
    return median(gaps)


def staged_family_summary(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    lines = [
        "\\begin{tabular}{@{}lrrrrrrrll@{}}",
        "\\toprule",
        "Family & Inst. & B\\&B cert. & B\\&B lim. & SCIP cert. & HiGHS cert. & "
        "B\\&B time & SCIP time & HiGHS time & Observation \\\\",
        "\\midrule",
    ]
    for family in FAMILY_ORDER:
        inst_keys = {run_key_without_method(r) for r in rows if r["family"] == family}
        if not inst_keys:
            continue
        opt = [r for r in rows if r["family"] == family and r["method"] == "global_bnb_cached_cutoff_ordered"]
        scip = [r for r in rows if r["family"] == family and r["method"] == "scip"]
        highs = [r for r in rows if r["family"] == family and r["method"] == "highs"]
        opt_cert = sum(1 for r in opt if r["status"] == "optimal")
        opt_lim = sum(1 for r in opt if is_limited(r["status"]))
        scip_cert = sum(1 for r in scip if r["status"] == "optimal")
        highs_cert = sum(1 for r in highs if r["status"] == "optimal")
        obs = "all certified" if opt_cert == len(opt) and opt else "--"
        if opt_lim:
            obs = "limits with gaps"
        gap = hullround_gap_for_family(rows, family)
        if math.isfinite(gap):
            obs += f"; HR gap {fmt_num(100.0 * gap, 3)}\\%"
        lines.append(
            f"{FAMILY_LABELS.get(family, family)} & {len(inst_keys)} & {opt_cert} & {opt_lim} & "
            f"{scip_cert} & {highs_cert} & {fmt_num(median(value(r, 'runtime_seconds') for r in opt))} & "
            f"{fmt_num(median(value(r, 'runtime_seconds') for r in scip))} & "
            f"{fmt_num(median(value(r, 'runtime_seconds') for r in highs))} & {obs} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_table(out_dir / "staged_family_summary.tex", lines)


def staged_unresolved_cases(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    unresolved = [
        r
        for r in rows
        if r["method"] != "hullround" and (is_limited(r["status"]) or r["status"] in {"error", "infeasible", "unavailable"})
    ]
    lines = [
        "\\begin{tabular}{@{}lrrrllrrrrr@{}}",
        "\\toprule",
        "Family & $n$ & $m$ & $\\Gamma$ & Mode & Method & Status & LB & UB & Rel. gap & Nodes \\\\",
        "\\midrule",
    ]
    for r in sorted(unresolved, key=lambda x: (x["family"], int(float(x["n"])), int(float(x["m"])), x["gamma_mode"], x["method"], x["seed"])):
        lines.append(
            f"{FAMILY_LABELS.get(r['family'], r['family'])} & {r['n']} & {r['m']} & {r['gamma']} & {r['gamma_mode']} & "
            f"{METHOD_LABELS.get(r['method'], r['method'])} & {r['status']} & "
            f"{fmt_num(value(r, 'global_lower_bound'))} & {fmt_num(value(r, 'global_upper_bound'))} & "
            f"{fmt_num(value(r, 'relative_gap'))} & {fmt_int(value(r, 'nodes_explored'))} \\\\"
        )
    if not unresolved:
        lines.append("\\multicolumn{11}{@{}l@{}}{No unresolved exact/MILP rows.} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    write_table(out_dir / "staged_unresolved_cases.tex", lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = read_rows(Path(args.input))
    out_dir = Path(args.output_dir)
    method_summary(rows, out_dir)
    family_summary(rows, out_dir)
    status_summary(rows, out_dir)
    staged_solver_summary(rows, out_dir)
    staged_family_summary(rows, out_dir)
    staged_unresolved_cases(rows, out_dir)
    print(f"Wrote TeX tables to {out_dir}")


if __name__ == "__main__":
    main()
