#!/usr/bin/env python3
"""Generate TeX summary tables for branching-strategy diagnostics."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_INPUT = Path("results/branching_diagnostics/branching_diagnostics.csv")
DEFAULT_OUTPUT = Path("paper_versions/v2/tables/branching_diagnostics")


def _finite(values: Iterable[object]) -> List[float]:
    vals: List[float] = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    return vals


def _median(values: Iterable[object]) -> str:
    vals = _finite(values)
    if not vals:
        return "--"
    return f"{statistics.median(vals):.4g}"


def _status_counts(rows: Sequence[Dict[str, str]]) -> tuple[int, int]:
    certified = sum(1 for r in rows if r.get("status") == "optimal")
    limited = sum(1 for r in rows if r.get("status") in {"time_limit", "node_limit", "not_run_time_limit", "not_run_node_limit"})
    return certified, limited


def _escape(text: object) -> str:
    return str(text).replace("_", "\\_")


def _table(headers: Sequence[str], body: Sequence[Sequence[object]], caption: str, label: str) -> str:
    cols = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in body:
        lines.append(" & ".join(_escape(x) for x in row) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_rule: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_rule.setdefault(row.get("branch_rule", ""), []).append(row)

    summary_rows: List[List[object]] = []
    for rule, rule_rows in sorted(by_rule.items()):
        certified, limited = _status_counts(rule_rows)
        summary_rows.append(
            [
                rule,
                len(rule_rows),
                certified,
                limited,
                _median(r.get("relative_gap") for r in rule_rows),
                _median(r.get("runtime_seconds") for r in rule_rows),
                _median(r.get("nodes_explored") for r in rule_rows),
                _median(r.get("nodes_pruned_bound") for r in rule_rows),
                _median(r.get("strong_branching_time") for r in rule_rows),
                "--" if rule != "strong_branching_lite" else "top-tree probing",
            ]
        )
    (output_dir / "branching_strategy_summary.tex").write_text(
        _table(
            [
                "Branch rule",
                "Rows",
                "Cert.",
                "Limited",
                "Med. gap",
                "Med. time",
                "Med. nodes",
                "Med. bound prunes",
                "Med. strong time",
                "Notes",
            ],
            summary_rows,
            "Tight-capacity branch-strategy diagnostic summary. Exact rows use the full theta set; limited rows report valid final gaps.",
            "tab:branching-strategy-summary",
        ),
        encoding="utf-8",
    )

    limited_rows = [r for r in rows if r.get("status") in {"time_limit", "node_limit"}]
    limited_body = [
        [
            r.get("n", ""),
            r.get("m", ""),
            r.get("gamma_mode", ""),
            r.get("seed", ""),
            r.get("branch_rule", ""),
            _median([r.get("relative_gap")]),
            _median([r.get("runtime_seconds")]),
            _median([r.get("nodes_explored")]),
            "node LP/search",
        ]
        for r in limited_rows
    ]
    (output_dir / "branching_limited_cases.tex").write_text(
        _table(
            ["n", "m", "Gamma", "Seed", "Branch rule", "Gap", "Time", "Nodes", "Bottleneck"],
            limited_body,
            "Limited tight-capacity rows in the branch-strategy diagnostic run.",
            "tab:branching-limited-cases",
        ),
        encoding="utf-8",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summarize(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
