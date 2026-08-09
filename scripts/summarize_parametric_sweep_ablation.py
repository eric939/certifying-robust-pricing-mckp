#!/usr/bin/env python3
"""Summarize parametric sweep construction/root-bound ablations."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CSV_NAME = "ablation_results.csv"


def read_rows(path: Path) -> List[Dict[str, str]]:
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


def fmt(x: float, digits: int = 4) -> str:
    if not math.isfinite(x):
        return "--"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{digits}f}"


def tex(text: object) -> str:
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def case_key(row: Dict[str, str]) -> Tuple[str, str, str, str, str]:
    return row["family"], row["n"], row["m"], row["gamma"], row["seed"]


def write_summary_csv(rows: Sequence[Dict[str, str]], path: Path) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["method"])].append(row)
    summary: List[Dict[str, object]] = []
    for (family, method), rs in sorted(grouped.items()):
        ok = [r for r in rs if r.get("status") == "ok"]
        summary.append(
            {
                "family": family,
                "method": method,
                "rows": len(rs),
                "failed_rows": len(rs) - len(ok),
                "median_total_runtime_seconds": median(value(r, "total_runtime_seconds") for r in ok),
                "median_theta_data_time_seconds": median(value(r, "theta_data_time_seconds") for r in ok),
                "median_baseline_cost_capacity_time_seconds": median(value(r, "baseline_cost_capacity_time_seconds") for r in ok),
                "median_hull_time_seconds": median(value(r, "hull_time_seconds") for r in ok),
                "median_root_lp_time_seconds": median(value(r, "root_lp_time_seconds") for r in ok),
                "median_hull_reuse_rate": median(value(r, "hull_reuse_rate") for r in ok),
                "max_validation_residual": max(
                    finite(
                        max(
                            value(r, "max_abs_s_theta_recompute_error"),
                            value(r, "max_abs_cost_recompute_error"),
                            value(r, "max_abs_capacity_recompute_error"),
                            value(r, "max_abs_root_bound_recompute_error"),
                        )
                        for r in ok
                    ),
                    default=float("nan"),
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if summary:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    return summary


def speedups(rows: Sequence[Dict[str, str]], target: str) -> Dict[str, List[float]]:
    by_case_method = {(case_key(r), r["method"]): r for r in rows if r.get("status") == "ok"}
    out: Dict[str, List[float]] = defaultdict(list)
    for (case, method), base in by_case_method.items():
        if method != "independent_recompute":
            continue
        other = by_case_method.get((case, target))
        if not other:
            continue
        base_t = value(base, "total_runtime_seconds")
        other_t = value(other, "total_runtime_seconds")
        if base_t > 0 and other_t > 0:
            out[case[0]].append(base_t / other_t)
    return out


def write_tex(summary: Sequence[Dict[str, object]], rows: Sequence[Dict[str, str]], tables_dir: Path, label: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{@{}llrrrrrr@{}}",
        "\\toprule",
        "Family & Method & Rows & Fail & Med. total & Med. hull & Med. root LP & Reuse \\\\",
        "\\midrule",
    ]
    for row in summary:
        lines.append(
            f"{tex(row['family'])} & {tex(row['method'])} & {row['rows']} & {row['failed_rows']} & "
            f"{fmt(float(row['median_total_runtime_seconds']))} & {fmt(float(row['median_hull_time_seconds']))} & "
            f"{fmt(float(row['median_root_lp_time_seconds']))} & {fmt(float(row['median_hull_reuse_rate']), 3)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / f"ablation_summary_{label}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sp_rebuild = speedups(rows, "incremental_sweep_rebuild")
    sp_reuse = speedups(rows, "incremental_sweep_safe_reuse")
    speed_lines = [
        "\\begin{tabular}{@{}lrr@{}}",
        "\\toprule",
        "Family & Recompute/rebuild & Recompute/safe reuse \\\\",
        "\\midrule",
    ]
    for family in sorted(set(sp_rebuild) | set(sp_reuse)):
        speed_lines.append(f"{tex(family)} & {fmt(median(sp_rebuild.get(family, [])), 3)} & {fmt(median(sp_reuse.get(family, [])), 3)} \\\\")
    speed_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / f"ablation_speedups_{label}.tex").write_text("\n".join(speed_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--label", default="smoke")
    args = parser.parse_args()
    rows = read_rows(args.input_dir / CSV_NAME)
    summary = write_summary_csv(rows, args.input_dir / "ablation_summary.csv")
    write_tex(summary, rows, args.tables_dir, args.label)
    print(f"Wrote {args.input_dir / 'ablation_summary.csv'}")
    print(f"Wrote {args.tables_dir / f'ablation_summary_{args.label}.tex'}")
    print(f"Wrote {args.tables_dir / f'ablation_speedups_{args.label}.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
