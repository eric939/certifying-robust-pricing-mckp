#!/usr/bin/env python3
"""Summarize segment-local Gamma-budget experiment outputs."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CSV_NAME = "local_budget_results.csv"


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


def b(row: Dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def fmt(x: float, digits: int = 3) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def tex(text: object) -> str:
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def write_summary(rows: Sequence[Dict[str, str]], path: Path) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["method"])].append(row)
    summary: List[Dict[str, object]] = []
    for (case, method), rs in sorted(grouped.items()):
        summary.append(
            {
                "case": case,
                "method": method,
                "rows": len(rs),
                "certified_rows": sum(1 for r in rs if b(r, "certified_optimal") or r.get("status") == "optimal"),
                "guard_rows": sum(1 for r in rs if b(r, "product_guard_triggered")),
                "bruteforce_parity_rows": sum(1 for r in rs if b(r, "bruteforce_parity_passed")),
                "one_segment_parity_rows": sum(1 for r in rs if b(r, "one_segment_parity_passed")),
                "median_theta_vector_count": median(value(r, "theta_vector_count_total") for r in rs),
                "median_runtime_seconds": median(value(r, "runtime_seconds") for r in rs),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if summary:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    return summary


def write_tex(summary: Sequence[Dict[str, object]], tables_dir: Path, label: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{@{}llrrrrrr@{}}",
        "\\toprule",
        "Case & Method & Rows & Cert. & Guard & Brute & One-seg. & Med. runtime \\\\",
        "\\midrule",
    ]
    for row in summary:
        lines.append(
            f"{tex(row['case'])} & {tex(row['method'])} & {row['rows']} & {row['certified_rows']} & {row['guard_rows']} & "
            f"{row['bruteforce_parity_rows']} & {row['one_segment_parity_rows']} & {fmt(float(row['median_runtime_seconds']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / f"segment_local_budget_{label}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--label", default="smoke")
    args = parser.parse_args()
    rows = read_rows(args.input_dir / CSV_NAME)
    summary = write_summary(rows, args.input_dir / "local_budget_summary.csv")
    write_tex(summary, args.tables_dir, args.label)
    print(f"Wrote {args.input_dir / 'local_budget_summary.csv'}")
    print(f"Wrote {args.tables_dir / f'segment_local_budget_{args.label}.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
