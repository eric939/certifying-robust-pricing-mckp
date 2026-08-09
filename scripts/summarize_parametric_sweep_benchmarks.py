#!/usr/bin/env python3
"""Summarize exact enumeration vs exact sweep benchmark CSV outputs."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CSV_NAME = "parametric_sweep_results.csv"


def read_rows(input_dir: Path) -> List[Dict[str, str]]:
    path = input_dir / CSV_NAME
    if not path.exists():
        path = input_dir / "parametric_sweep_benchmarks.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def value(row: Dict[str, str], key: str) -> float:
    try:
        x = float(row.get(key, "nan"))
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def median(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(statistics.median(vals)) if vals else float("nan")


def fmt(x: float, digits: int = 3) -> str:
    if not math.isfinite(x):
        return "--"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{digits}f}"


def tex(text: object) -> str:
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def comparable_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    return row["family"], row["n"], row["m"], row["gamma"], row["seed"]


def bool_field(row: Dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def write_csv_summary(rows: Sequence[Dict[str, str]], output_path: Path) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["method"])].append(row)
    summary: List[Dict[str, object]] = []
    for (family, method), rs in sorted(grouped.items()):
        certified = [r for r in rs if bool_field(r, "certified_optimal") or r.get("status") == "optimal"]
        feasible = [r for r in rs if bool_field(r, "feasible_incumbent") or r.get("status") == "feasible"]
        limited = [r for r in rs if r.get("status") in {"time_limit", "node_limit", "limited"}]
        summary.append(
            {
                "family": family,
                "method": method,
                "rows": len(rs),
                "certified_rows": len(certified),
                "feasible_rows": len(feasible),
                "limited_rows": len(limited),
                "median_runtime_seconds": median(value(r, "runtime_seconds") for r in certified or rs),
                "median_relative_gap_limited": median(value(r, "rel_gap") for r in limited),
                "max_relative_gap_limited": max(finite(value(r, "rel_gap") for r in limited), default=float("nan")),
                "median_theta_evaluated": median(value(r, "theta_count_evaluated") for r in rs),
                "median_theta_pruned": median(value(r, "theta_count_pruned") for r in rs),
                "median_hull_rebuilds": median(value(r, "hull_rebuilds_total") for r in rs),
                "median_hull_reuses": median(value(r, "hull_reuses_total") for r in rs),
                "median_hull_reuse_rate": median(value(r, "hull_reuse_rate") for r in rs),
                "max_validation_error": max(finite(value(r, "max_abs_validation_error") for r in rs), default=float("nan")),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    return summary


def write_solver_summary(summary: Sequence[Dict[str, object]], tables_dir: Path, label: str) -> None:
    lines = [
        "\\begin{tabular}{@{}llrrrrrr@{}}",
        "\\toprule",
        "Family & Method & Rows & Cert. & Feas. & Limited & Med. time (s) & Reuse \\\\",
        "\\midrule",
    ]
    for row in summary:
        lines.append(
            f"{tex(row['family'])} & {tex(row['method'])} & {row['rows']} & {row['certified_rows']} & {row['feasible_rows']} & "
            f"{row['limited_rows']} & {fmt(float(row['median_runtime_seconds']))} & "
            f"{fmt(float(row['median_hull_reuse_rate']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / f"solver_summary_{label}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_speedup_table(rows: Sequence[Dict[str, str]], tables_dir: Path, label: str) -> None:
    by_key_method = {(comparable_key(r), r["method"]): r for r in rows}
    speedups: Dict[str, List[float]] = defaultdict(list)
    for (case, method), enum_row in by_key_method.items():
        if method != "exact_enum_current":
            continue
        sweep = by_key_method.get((case, "exact_sweep_new"))
        if not sweep:
            continue
        if enum_row.get("status") == "optimal" and sweep.get("status") == "optimal":
            enum_t = value(enum_row, "runtime_seconds")
            sweep_t = value(sweep, "runtime_seconds")
            if enum_t > 0 and sweep_t > 0:
                speedups[case[0]].append(enum_t / sweep_t)
    lines = [
        "\\begin{tabular}{@{}lrr@{}}",
        "\\toprule",
        "Family & Matched rows & Median enum/sweep runtime \\\\",
        "\\midrule",
    ]
    for family, vals in sorted(speedups.items()):
        lines.append(f"{tex(family)} & {len(vals)} & {fmt(median(vals))} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / f"solver_speedup_{label}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_availability_table(rows: Sequence[Dict[str, str]], tables_dir: Path, label: str) -> None:
    methods = ["scipy_highs", "scip", "gurobi", "cplex"]
    lines = [
        "\\begin{tabular}{@{}lrl@{}}",
        "\\toprule",
        "Backend & Available rows & Message \\\\",
        "\\midrule",
    ]
    for method in methods:
        rs = [r for r in rows if r["method"] == method]
        if not rs:
            continue
        available = sum(1 for r in rs if bool_field(r, "backend_available"))
        message = next((r.get("backend_message", "") for r in rs if r.get("backend_message", "")), "")
        message = message.replace("theta-decomposition fixed-theta", "$\\theta$-decomposition fixed-$\\theta$")
        lines.append(f"{tex(method)} & {available}/{len(rs)} & {tex(message)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / f"solver_availability_{label}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--label", default="smoke")
    args = parser.parse_args()
    rows = read_rows(args.input_dir)
    summary = write_csv_summary(rows, args.input_dir / "parametric_sweep_summary.csv")
    write_solver_summary(summary, args.tables_dir, args.label)
    write_speedup_table(rows, args.tables_dir, args.label)
    write_availability_table(rows, args.tables_dir, args.label)
    print(f"Wrote {args.input_dir / 'parametric_sweep_summary.csv'}")
    print(f"Wrote {args.tables_dir / f'solver_summary_{args.label}.tex'}")
    print(f"Wrote {args.tables_dir / f'solver_speedup_{args.label}.tex'}")
    print(f"Wrote {args.tables_dir / f'solver_availability_{args.label}.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
