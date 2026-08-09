#!/usr/bin/env python3
"""Generate focused TeX tables for the final Path A publication experiment."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


METHOD_LABELS = {
    "hullround": "HullRound",
    "global_bnb_cached_cutoff_ordered": "Exact $\\theta$-B\\&B",
    "scip": "SCIP",
    "highs": "HiGHS",
}
METHOD_ORDER = ["hullround", "global_bnb_cached_cutoff_ordered", "scip", "highs"]
FAMILY_LABELS = {
    "economic": "Economic",
    "many_theta": "Many $\\theta$",
    "hull_compression": "Hull compression",
    "adversarial": "Low compression",
}
FAMILY_ORDER = ["economic", "many_theta", "hull_compression", "adversarial"]


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def val(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


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


def pct(x: float, digits: int = 4) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{100.0 * x:.{digits}f}\\%"


def key(row: Dict[str, str]) -> Tuple[str, str, str, str, str, str]:
    return (row["family"], row["n"], row["m"], row["gamma"], row["gamma_mode"], row["seed"])


def escape(text: object) -> str:
    return str(text).replace("_", "\\_")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def table(headers: Sequence[str], rows: Sequence[Sequence[object]], *, align: str) -> str:
    lines = [f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule"]
    lines.extend(" & ".join(escape(x) for x in row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def hullround_gaps(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str, str, str, str, str], float]:
    hull = {key(r): r for r in rows if r["method"] == "hullround"}
    gaps: Dict[Tuple[str, str, str, str, str, str], float] = {}
    for row in rows:
        if row["method"] != "global_bnb_cached_cutoff_ordered" or row["status"] != "optimal":
            continue
        h = hull.get(key(row))
        if h is None:
            continue
        opt = val(row, "objective")
        hv = val(h, "objective")
        if opt > 0 and math.isfinite(hv):
            gaps[key(row)] = max(0.0, (opt - hv) / opt)
    return gaps


def quality_summary(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    gaps = hullround_gaps(rows)
    body: List[List[object]] = []
    for family in FAMILY_ORDER:
        inst_keys = {key(r) for r in rows if r["family"] == family}
        exact = [r for r in rows if r["family"] == family and r["method"] == "global_bnb_cached_cutoff_ordered"]
        hull = [r for r in rows if r["family"] == family and r["method"] == "hullround"]
        scip = [r for r in rows if r["family"] == family and r["method"] == "scip"]
        highs = [r for r in rows if r["family"] == family and r["method"] == "highs"]
        fam_gaps = [g for k, g in gaps.items() if k[0] == family]
        obs = "exact match" if fam_gaps and max(fam_gaps) <= 1e-12 else "near-exact"
        body.append(
            [
                FAMILY_LABELS.get(family, family),
                len(inst_keys),
                sum(1 for r in exact if r["status"] == "optimal"),
                sum(1 for r in hull if r["status"] == "feasible" and r.get("valid_certificate", "").lower() == "true"),
                pct(median(fam_gaps), 5),
                pct(max(fam_gaps) if fam_gaps else float("nan"), 5),
                fmt(median(val(r, "runtime_seconds") for r in hull)),
                fmt(median(val(r, "runtime_seconds") for r in exact)),
                f"{sum(1 for r in scip if r['status']=='optimal')}/{sum(1 for r in highs if r['status']=='optimal')}",
                obs,
            ]
        )
    write(
        out_dir / "final_hullround_quality_summary.tex",
        table(
            [
                "Family",
                "Inst.",
                "Exact cert.",
                "HR feas.",
                "Med. HR gap",
                "Max HR gap",
                "HR time",
                "Exact time",
                "SCIP/HiGHS cert.",
                "Observation",
            ],
            body,
            align="lrrrrrrrll",
        ),
    )


def solver_status_summary(rows: Sequence[Dict[str, str]], out_dir: Path) -> None:
    body: List[List[object]] = []
    for method in METHOD_ORDER:
        rs = [r for r in rows if r["method"] == method]
        c = Counter(r["status"] for r in rs)
        cert = c["feasible"] if method == "hullround" else c["optimal"]
        limited = c["time_limit"] + c["node_limit"] + c["timelimit"] + c["userinterrupt"]
        gaps = [val(r, "relative_gap") for r in rs if math.isfinite(val(r, "relative_gap"))]
        note = "heuristic" if method == "hullround" else "certified"
        body.append(
            [
                METHOD_LABELS.get(method, method),
                len(rs),
                cert,
                limited,
                fmt(median(val(r, "runtime_seconds") for r in rs)),
                fmt(median(gaps)),
                note,
            ]
        )
    write(
        out_dir / "final_solver_status_summary.tex",
        table(
            ["Method", "Rows", "Cert./feas.", "Limited", "Median runtime", "Median gap", "Notes"],
            body,
            align="lrrrrrl",
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    rows = read_rows(Path(args.input))
    out_dir = Path(args.output_dir)
    quality_summary(rows, out_dir)
    solver_status_summary(rows, out_dir)
    print(f"Wrote final experiment TeX tables to {out_dir}")


if __name__ == "__main__":
    main()
