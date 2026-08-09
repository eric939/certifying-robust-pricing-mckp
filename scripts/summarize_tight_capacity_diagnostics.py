#!/usr/bin/env python3
"""Generate TeX tables for tight-capacity B&B diagnostics."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def _median(values: Iterable[float]) -> str:
    vals = _finite(values)
    if not vals:
        return "--"
    value = float(statistics.median(vals))
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.3g}"


def _tex_escape(text: object) -> str:
    return str(text).replace("_", "\\_")


def _write_table(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    cols = "l" * len(header)
    lines = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_tex_escape(cell) for cell in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _status_limited(status: str) -> bool:
    return status in {"time_limit", "node_limit", "not_run_time_limit", "not_run_node_limit"}


def diagnostic_summary(rows: List[Dict[str, str]]) -> List[List[object]]:
    exact = [r for r in rows if r.get("method", "").startswith("global_bnb_cached_cutoff_ordered")]
    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in exact:
        groups[(row.get("method", ""), row.get("n", ""), row.get("m", ""), row.get("gamma_mode", ""))].append(row)
    out: List[List[object]] = []
    for (method, n, m, gamma_mode), group in sorted(groups.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), int(kv[0][2]), kv[0][3])):
        certified = sum(1 for r in group if r.get("status") == "optimal")
        limited = sum(1 for r in group if _status_limited(r.get("status", "")))
        bound_prunes = [_float(r, "total_nodes_pruned_bound") for r in group]
        cutoff_prunes = [_float(r, "total_nodes_pruned_cutoff") for r in group]
        total_nodes = [_float(r, "total_nodes_explored") + _float(r, "total_nodes_pruned_bound") + _float(r, "total_nodes_pruned_infeasible") for r in group]
        bound_rates = [b / t for b, t in zip(bound_prunes, total_nodes) if math.isfinite(b) and math.isfinite(t) and t > 0]
        cutoff_rates = [c / t for c, t in zip(cutoff_prunes, total_nodes) if math.isfinite(c) and math.isfinite(t) and t > 0]
        node_lp_share = [
            _float(r, "time_node_lp_total") / _float(r, "runtime_seconds")
            for r in group
            if _float(r, "runtime_seconds") > 0 and math.isfinite(_float(r, "time_node_lp_total"))
        ]
        out.append(
            [
                method.replace("global_bnb_cached_cutoff_ordered_", ""),
                f"{n}/{m}/{gamma_mode}",
                len(group),
                certified,
                limited,
                _median(_float(r, "relative_gap") for r in group),
                _median(_float(r, "total_nodes_explored") for r in group),
                _median(_float(r, "runtime_seconds") for r in group),
                _median(node_lp_share),
                _median(bound_rates),
                _median(cutoff_rates),
                _median(_float(r, "hardest_theta_gap") for r in group),
            ]
        )
    return out


def unresolved_summary(rows: List[Dict[str, str]]) -> List[List[object]]:
    exact = [r for r in rows if r.get("method", "").startswith("global_bnb_cached_cutoff_ordered") and _status_limited(r.get("status", ""))]
    out: List[List[object]] = []
    for row in sorted(exact, key=lambda r: (int(r.get("n", 0)), int(r.get("m", 0)), r.get("gamma_mode", ""), int(r.get("seed", 0)))):
        components = {
            "root LP": _float(row, "time_root_lp_total"),
            "node LP": _float(row, "time_node_lp_total"),
            "hull": _float(row, "time_hull_build_total"),
            "greedy": _float(row, "time_greedy_lp_total"),
            "branch": _float(row, "time_branching_total"),
            "child": _float(row, "time_child_generation_total"),
        }
        dominant = max(components, key=lambda key: -1.0 if not math.isfinite(components[key]) else components[key])
        out.append(
            [
                row.get("n", ""),
                row.get("m", ""),
                row.get("gamma_mode", ""),
                row.get("seed", ""),
                row.get("method", "").replace("global_bnb_cached_cutoff_ordered_", ""),
                _median([_float(row, "relative_gap")]),
                _median([_float(row, "total_nodes_explored")]),
                _median([_float(row, "runtime_seconds")]),
                row.get("unresolved_theta_count", "--"),
                _median([_float(row, "hardest_theta")]),
                _median([_float(row, "hardest_theta_gap")]),
                dominant,
            ]
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/tight_capacity_diagnostics/tight_capacity_diagnostics.csv")
    parser.add_argument("--output-dir", default="paper_versions/v2/tables/tight_capacity_diagnostics")
    args = parser.parse_args(argv)
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"missing input CSV: {in_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with in_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary_rows = diagnostic_summary(rows)
    unresolved_rows = unresolved_summary(rows)
    _write_table(
        out_dir / "tight_capacity_diagnostic_summary.tex",
        [
            "Method",
            "n/m/$\\Gamma$",
            "Instances",
            "Certified",
            "Limited",
            "Med. rel. gap",
            "Med. nodes",
            "Med. time",
            "Med. node-LP share",
            "Med. bound prune",
            "Med. cutoff prune",
            "Hardest $\\theta$ gap",
        ],
        summary_rows,
    )
    _write_table(
        out_dir / "tight_capacity_unresolved_summary.tex",
        [
            "$n$",
            "$m$",
            "$\\Gamma$ mode",
            "Seed",
            "Method",
            "Rel. gap",
            "Nodes",
            "Runtime",
            "Unres. $\\theta$",
            "Hardest $\\theta$",
            "Hardest gap",
            "Dominant time",
        ],
        unresolved_rows,
    )
    _write_table(
        out_dir / "fastbound_summary.tex",
        [
            "Method",
            "n/m/$\\Gamma$",
            "Instances",
            "Certified",
            "Limited",
            "Med. rel. gap",
            "Med. nodes",
            "Med. time",
            "Med. node-LP share",
            "Med. bound prune",
            "Med. cutoff prune",
            "Hardest $\\theta$ gap",
        ],
        summary_rows,
    )
    _write_table(
        out_dir / "fastbound_limited_cases.tex",
        [
            "$n$",
            "$m$",
            "$\\Gamma$ mode",
            "Seed",
            "Method",
            "Rel. gap",
            "Nodes",
            "Runtime",
            "Unres. $\\theta$",
            "Hardest $\\theta$",
            "Hardest gap",
            "Dominant time",
        ],
        unresolved_rows,
    )


if __name__ == "__main__":
    main()
