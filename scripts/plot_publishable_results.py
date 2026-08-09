#!/usr/bin/env python3
"""Regenerate publishable figures, tables, and auto-number macros from CSVs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.run_publishable_experiments import (  # noqa: E402
    CSV_DIR,
    LOG_DIR,
    plot_frontier,
    plot_gap,
    plot_retail,
    plot_scalability,
)
from scripts.run_solver_benchmarks import (  # noqa: E402
    plot_solver_figures,
    update_auto_numbers,
    write_tables,
)


def read_csv(name: str):
    with (CSV_DIR / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    availability_path = LOG_DIR / "solver_availability.json"
    availability = json.loads(availability_path.read_text(encoding="utf-8")) if availability_path.exists() else {}

    gap_rows = read_csv("synthetic_gap_replicates.csv")
    exact_rows = read_csv("solver_benchmark_exact.csv")
    solver_scale_rows = read_csv("solver_benchmark_scalability.csv")
    scale_rows = read_csv("scalability.csv")
    cert_rows = read_csv("synthetic_frontier_certificates.csv")
    mc_rows = read_csv("synthetic_stress_mc.csv")
    retail_rows = read_csv("retail_replicates.csv")
    retail_seg = read_csv("retail_segment_adjustments.csv")
    retail_conc = read_csv("retail_margin_concentration.csv")

    plot_gap(gap_rows)
    plot_solver_figures(exact_rows, solver_scale_rows)
    plot_frontier(cert_rows, mc_rows)
    plot_scalability(scale_rows)
    plot_retail(retail_rows, retail_seg, retail_conc)
    write_tables(exact_rows, solver_scale_rows)
    update_auto_numbers(availability)
    print("publishable figures, tables, and macros regenerated")


if __name__ == "__main__":
    main()
