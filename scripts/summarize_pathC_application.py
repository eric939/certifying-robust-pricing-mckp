#!/usr/bin/env python3
"""Generate LaTeX tables for the Path C application."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finite(vals: Iterable[object]) -> List[float]:
    out = []
    for v in vals:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def med(vals: Iterable[object]) -> float:
    xs = finite(vals)
    return float(statistics.median(xs)) if xs else float("nan")


def maxf(vals: Iterable[object]) -> float:
    xs = finite(vals)
    return max(xs) if xs else float("nan")


def fmt(x: float, digits: int = 3) -> str:
    if not math.isfinite(float(x)):
        return "--"
    if abs(x) >= 1000:
        return f"{x:.1f}"
    return f"{x:.{digits}f}"


def tex_escape(s: str) -> str:
    return s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


def write_table(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = "l" + "r" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/pathC/semisynthetic_application")
    parser.add_argument("--output-dir", default="paper_versions/v2/tables/pathC_application")
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy = read_csv(inp / "pathC_policy_results.csv")
    stress = read_csv(inp / "pathC_stress_results.csv")
    segment = read_csv(inp / "pathC_segment_diagnostics.csv")
    exact = read_csv(inp / "pathC_exact_subset_results.csv")

    # Frontier table for HullRound by gamma.
    stress_by = defaultdict(list)
    for r in stress:
        if r.get("method") == "HullRound":
            stress_by[(int(float(r["eval_gamma"])), r["protocol"])].append(r)
    rows = []
    by_gamma = defaultdict(list)
    for r in policy:
        if r.get("method") == "HullRound":
            by_gamma[int(float(r["eval_gamma"]))].append(r)
    for gamma in sorted(by_gamma):
        group = by_gamma[gamma]
        rows.append(
            [
                str(gamma),
                fmt(med(r["revenue_ratio"] for r in group), 4),
                fmt(med(r["robust_certificate"] for r in group), 1),
                fmt(med(r["violation_probability"] for r in stress_by[(gamma, "iid")]), 3),
                fmt(med(r["violation_probability"] for r in stress_by[(gamma, "segment_block")]), 3),
                fmt(med(r["violation_probability"] for r in stress_by[(gamma, "heavy_tail")]), 3),
                fmt(med(r["share_changed_vs_nominal"] for r in group), 3),
                fmt(med(r["avg_abs_price_change"] for r in group), 3),
            ]
        )
    write_table(
        out / "pathC_frontier_summary.tex",
        ["$\\Gamma$", "Rev.", "Cert.", "IID viol.", "Block viol.", "Heavy viol.", "Changed", "$|\\Delta p|$"],
        rows,
    )

    # Segment summary at sqrt gamma (or nearest nonzero gamma).
    hull_gammas = sorted(by_gamma)
    target_gamma = min((g for g in hull_gammas if g > 0), default=0, key=lambda g: abs(g - math.sqrt(max(hull_gammas) if hull_gammas else 1)))
    seg_groups = defaultdict(list)
    for r in segment:
        if r.get("method") == "HullRound" and int(float(r["eval_gamma"])) == target_gamma:
            seg_groups[r["segment"]].append(r)
    rows = []
    for seg in sorted(seg_groups):
        group = seg_groups[seg]
        rows.append(
            [
                tex_escape(seg),
                fmt(med(r["median_signed_price_change"] for r in group), 3),
                fmt(med(r["share_changed"] for r in group), 3),
                fmt(med(r["median_uncertainty_scale"] for r in group), 3),
                fmt(med(r["margin_contribution_change"] for r in group), 1),
            ]
        )
    write_table(
        out / "pathC_segment_summary.tex",
        ["Segment", "Median $\\Delta p$", "Share changed", "Uncert.", "$\\Delta$ margin"],
        rows,
    )

    # Exact subset summary.
    method_groups = defaultdict(list)
    for r in exact:
        method_groups[r["method"]].append(r)
    rows = []
    method_labels = {
        "Global theta B&B": "Global $\\theta$-B\\&B",
    }
    for method in sorted(method_groups):
        group = method_groups[method]
        certified = sum(1 for r in group if r.get("status") == "optimal")
        feasible = sum(1 for r in group if r.get("status") in {"feasible", "optimal"})
        gaps = [float(r["gap_to_exact"]) for r in group if r.get("gap_to_exact") not in {"", "nan"} and math.isfinite(float(r["gap_to_exact"]))]
        rows.append(
            [
                method_labels.get(method, tex_escape(method)),
                str(certified),
                str(feasible),
                fmt(med(gaps), 6),
                fmt(maxf(gaps), 6),
                fmt(med(r["runtime_seconds"] for r in group), 3),
            ]
        )
    write_table(
        out / "pathC_exact_subset_summary.tex",
        ["Method", "Exact cert.", "Feasible", "Med. gap", "Max gap", "Med. time"],
        rows,
    )
    print(f"Wrote Path C tables to {out}")


if __name__ == "__main__":
    main()
