#!/usr/bin/env python3
"""Build a parametric-sweep-specific claim/evidence table."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


def exists(rel: str) -> str:
    return "yes" if (ROOT / rel).exists() else "missing"


def exists_any(*rels: str) -> str:
    return "yes" if any((ROOT / rel).exists() for rel in rels) else "missing"


def rows() -> List[Dict[str, str]]:
    return [
        {
            "claim": "Exact sweep evaluates the same full breakpoint set as enumeration.",
            "evidence": "tests/test_parametric_theta_sweep.py; results/parametric_sweep_ablation_smoke/ablation_results.csv",
            "status": exists("tests/test_parametric_theta_sweep.py"),
            "limitation": "No reduced-$\\theta$ exactness claim is made.",
        },
        {
            "claim": "Incremental $\\theta$ update preserves fixed-$\\theta$ data.",
            "evidence": "focused tests; ablation residual columns",
            "status": exists_any(
                "results/parametric_sweep_ablation_smoke/ablation_results.csv",
                "results/top_journal_strengthening/parametric_sweep_ablation_smoke/ablation_results.csv",
            ),
            "limitation": "Numerical tolerance applies.",
        },
        {
            "claim": "Safe hull maintenance is exact.",
            "evidence": "hull reuse/rebuild tests; max root-bound residuals in ablation CSV",
            "status": exists_any(
                "results/parametric_sweep_ablation_smoke/ablation_results.csv",
                "results/top_journal_strengthening/parametric_sweep_ablation_smoke/ablation_results.csv",
            ),
            "limitation": "Reuse may be rare when point geometry changes.",
        },
        {
            "claim": "Exact sweep solver has the same optimum/gap semantics as enumeration.",
            "evidence": "sweep-vs-enumeration tests; solver benchmark matched certified rows",
            "status": exists_any(
                "results/parametric_sweep_smoke/parametric_sweep_results.csv",
                "results/top_journal_strengthening/parametric_sweep_smoke/parametric_sweep_results.csv",
            ),
            "limitation": "Solver runtime can still be dominated by B&B search.",
        },
        {
            "claim": "Segment-local Gamma budgets are exact on small $\\theta$-vector products.",
            "evidence": "tests/test_segment_local_budgets.py; segment-local smoke CSV",
            "status": exists_any(
                "results/segment_local_budget_smoke/local_budget_results.csv",
                "results/top_journal_strengthening/segment_local_budget_smoke/local_budget_results.csv",
            ),
            "limitation": "$\\theta$-vector products can grow quickly and are guarded.",
        },
        {
            "claim": "Many-theta and tight-capacity benchmark evidence is limited to generated rows.",
            "evidence": "solver summary tables and CSVs",
            "status": exists_any(
                "paper_versions/v2/tables/parametric_sweep/solver_summary_smoke.tex",
                "paper_versions/v2/tables/top_journal_strengthening/solver_summary_smoke.tex",
            ),
            "limitation": "No universal speed-superiority claim.",
        },
        {
            "claim": "Optional solver comparisons depend on availability.",
            "evidence": "solver availability tables",
            "status": exists_any(
                "paper_versions/v2/tables/parametric_sweep/solver_availability_smoke.tex",
                "paper_versions/v2/tables/top_journal_strengthening/solver_availability_smoke.tex",
            ),
            "limitation": "Gurobi/CPLEX require local licenses.",
        },
    ]


def tex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("&", "\\&")


def write_csv(path: Path, data: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def write_tex(path: Path, data: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{@{}p{0.27\\linewidth}p{0.31\\linewidth}p{0.12\\linewidth}p{0.23\\linewidth}@{}}",
        "\\toprule",
        "Claim & Evidence & Status & Limitation \\\\",
        "\\midrule",
    ]
    for row in data:
        lines.append(
            f"{tex_escape(row['claim'])} & {tex_escape(row['evidence'])} & {tex_escape(row['status'])} & {tex_escape(row['limitation'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results" / "parametric_sweep_next_pass" / "claim_evidence_parametric_sweep.csv")
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=ROOT / "paper_versions" / "v2" / "tables" / "parametric_sweep" / "claim_evidence_parametric_sweep.tex",
    )
    args = parser.parse_args()
    data = rows()
    write_csv(args.output_csv, data)
    write_tex(args.output_tex, data)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
