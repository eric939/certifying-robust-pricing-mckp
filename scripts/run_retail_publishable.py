#!/usr/bin/env python3
"""Run the stylized retail experiment without rerunning all synthetic studies."""
from __future__ import annotations

import argparse
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
    choose_design,
    ensure_dirs,
    plot_retail,
    run_retail,
)
from scripts.run_solver_benchmarks import update_auto_numbers  # noqa: E402


def read_csv(name: str):
    with (CSV_DIR / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    design = choose_design(args.smoke)
    rows, seg_rows, conc_rows = run_retail(design)
    plot_retail(rows, seg_rows, conc_rows)
    availability_path = LOG_DIR / "solver_availability.json"
    availability = json.loads(availability_path.read_text(encoding="utf-8")) if availability_path.exists() else {}
    update_auto_numbers(availability)
    print(
        "retail experiment complete: "
        f"{len({r['seed'] for r in rows})} seeds, "
        f"{len({int(r['gamma']) for r in rows})} gamma values"
    )


if __name__ == "__main__":
    main()
