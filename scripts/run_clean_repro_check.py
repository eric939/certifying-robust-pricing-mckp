#!/usr/bin/env python3
"""Run a bounded code-only reproducibility check."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "clean_repro_check"
SUMMARY = OUT_DIR / "clean_repro_check_summary.txt"

KEY_SCRIPTS = [
    "scripts/run_publication_benchmarks.py",
    "scripts/summarize_publication_benchmarks.py",
    "scripts/plot_publication_benchmarks.py",
    "scripts/run_publishable_experiments.py",
    "scripts/run_solver_benchmarks.py",
    "scripts/plot_publishable_results.py",
    "scripts/run_pathC_data_calibration.py",
    "scripts/run_pathC_semisynthetic_application.py",
    "scripts/summarize_pathC_application.py",
    "scripts/plot_pathC_application.py",
    "scripts/summarize_empirical_synthesis.py",
    "scripts/plot_empirical_synthesis.py",
    "scripts/run_branching_strategy_diagnostics.py",
    "scripts/summarize_branching_strategy_diagnostics.py",
    "scripts/plot_branching_strategy_diagnostics.py",
    "scripts/run_tight_capacity_diagnostics.py",
    "scripts/summarize_tight_capacity_diagnostics.py",
    "scripts/plot_tight_capacity_diagnostics.py",
    "scripts/run_fixed_theta_bnb_benchmarks.py",
    "scripts/run_global_theta_bnb_benchmarks.py",
    "scripts/run_exact_bnb_performance_benchmarks.py",
    "scripts/run_parametric_sweep_ablation.py",
    "scripts/summarize_parametric_sweep_ablation.py",
    "scripts/run_parametric_sweep_benchmarks.py",
    "scripts/summarize_parametric_sweep_benchmarks.py",
    "scripts/run_segment_local_budget_experiments.py",
    "scripts/summarize_segment_local_budget_experiments.py",
    "scripts/build_parametric_sweep_claim_audit.py",
    "scripts/run_retail_publishable.py",
    "scripts/run_extended_publishable_experiments.py",
    "scripts/run_true_clean_room_check.py",
]


def log(message: str) -> None:
    print(message, flush=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    log("$ " + " ".join(cmd))
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, env=env)
    if proc.stdout:
        for line in proc.stdout.rstrip().splitlines():
            log("  " + line)
    if proc.stderr:
        for line in proc.stderr.rstrip().splitlines():
            log("  [stderr] " + line)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run bounded smoke checks.")
    args = parser.parse_args()
    if not args.quick:
        parser.error("Only --quick mode is currently implemented.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("Clean code-only reproducibility check\n", encoding="utf-8")

    try:
        log("1. Import check")
        run([sys.executable, "-c", "import robust_mckp; print('robust_mckp import ok')"])

        log("2. Unit tests")
        run([sys.executable, "-m", "pytest", "-q"])

        log("3. Py-compile experiment scripts")
        run([sys.executable, "-m", "py_compile", *KEY_SCRIPTS])

        log("4. Run synthetic benchmark smoke")
        run(
            [
                sys.executable,
                "scripts/run_publication_benchmarks.py",
                "--smoke",
                "--output-dir",
                "results/clean_repro_check/publication_benchmarks_smoke",
                "--time-limit",
                "1",
                "--node-limit",
                "2000",
            ]
        )

        log("5. Run parametric sweep ablation smoke")
        run(
            [
                sys.executable,
                "scripts/run_parametric_sweep_ablation.py",
                "--smoke",
                "--output-dir",
                "results/clean_repro_check/parametric_sweep_ablation_smoke",
                "--validate",
            ]
        )
        run(
            [
                sys.executable,
                "scripts/summarize_parametric_sweep_ablation.py",
                "--input-dir",
                "results/clean_repro_check/parametric_sweep_ablation_smoke",
                "--tables-dir",
                "results/clean_repro_check/generated_tables/parametric_sweep",
                "--label",
                "smoke",
            ]
        )

        log("6. Run parametric sweep solver smoke")
        run(
            [
                sys.executable,
                "scripts/run_parametric_sweep_benchmarks.py",
                "--smoke",
                "--output-dir",
                "results/clean_repro_check/parametric_sweep_smoke",
                "--time-limit",
                "2",
                "--node-limit",
                "5000",
                "--validate-sweep-sampled",
            ]
        )
        run(
            [
                sys.executable,
                "scripts/summarize_parametric_sweep_benchmarks.py",
                "--input-dir",
                "results/clean_repro_check/parametric_sweep_smoke",
                "--tables-dir",
                "results/clean_repro_check/generated_tables/parametric_sweep",
                "--label",
                "smoke",
            ]
        )

        log("7. Run segment-local budget smoke")
        run(
            [
                sys.executable,
                "scripts/run_segment_local_budget_experiments.py",
                "--smoke",
                "--output-dir",
                "results/clean_repro_check/segment_local_budget_smoke",
            ]
        )
        run(
            [
                sys.executable,
                "scripts/summarize_segment_local_budget_experiments.py",
                "--input-dir",
                "results/clean_repro_check/segment_local_budget_smoke",
                "--tables-dir",
                "results/clean_repro_check/generated_tables/parametric_sweep",
                "--label",
                "smoke",
            ]
        )

        log("8. Generate synthetic Path C calibration")
        run(
            [
                sys.executable,
                "scripts/run_pathC_data_calibration.py",
                "--source",
                "synthetic_only",
                "--output-dir",
                "results/clean_repro_check/pathC_calibration",
            ]
        )

        log("9. Run bounded Path C smoke")
        run(
            [
                sys.executable,
                "scripts/run_pathC_semisynthetic_application.py",
                "--calibration-dir",
                "results/clean_repro_check/pathC_calibration",
                "--output-dir",
                "results/clean_repro_check/pathC_smoke",
                "--seeds",
                "1",
                "--n",
                "60",
                "--m",
                "8",
                "--stress-scenarios",
                "200",
                "--gamma-grid",
                "0,sqrt,n",
                "--run-exact-small-subset",
                "--exact-time-limit",
                "2",
                "--exact-node-limit",
                "5000",
            ]
        )

        log("RESULT clean reproducibility check passed")
        return 0
    except Exception as exc:
        log(f"RESULT clean reproducibility check failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
