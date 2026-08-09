#!/usr/bin/env python3
"""Run the bounded reproducibility check in a fresh local copy."""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = Path("/tmp/robust_mckp_clean_check")

EXCLUDE_NAMES = {
    ".git",
    ".venv",
    ".venv_clean",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "results",
    "paper",
    "paper_current",
    "paper_notes",
    "paper_versions",
    "review_bundles",
    "submission_packages",
    "data_cache",
    "docs_archive",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_PATTERNS = {
    "* 2.py",
    "audit_v2_claims_against_results.py",
    "check_reported_numbers.py",
    "create_*package*.py",
    "create_final_*bundle.py",
    "create_submission_gate_packages.py",
    "plot_journal_composite_figures.py",
}


class Runner:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.summary = repo / "results" / "true_clean_room_check" / "true_clean_room_check_summary.txt"
        self.summary.parent.mkdir(parents=True, exist_ok=True)
        self.summary.write_text("True clean-room code-only reproducibility check\n", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.summary.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    def run(self, cmd: list[str], *, cwd: Path | None = None) -> None:
        cwd = cwd or self.repo
        self.log("$ " + " ".join(cmd))
        start = time.perf_counter()
        proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
        elapsed = time.perf_counter() - start
        if proc.stdout:
            for line in proc.stdout.rstrip().splitlines():
                self.log("  " + line)
        if proc.stderr:
            for line in proc.stderr.rstrip().splitlines():
                self.log("  [stderr] " + line)
        self.log(f"  [elapsed] {elapsed:.2f}s")
        if proc.returncode != 0:
            raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def ignore_func(_: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if (
            name in EXCLUDE_NAMES
            or any(name.endswith(suffix) for suffix in EXCLUDE_SUFFIXES)
            or any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDE_PATTERNS)
        ):
            ignored.add(name)
    return ignored


def copy_repo(work_dir: Path) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    dest = work_dir / "repo"
    shutil.copytree(ROOT, dest, ignore=ignore_func)
    return dest


def venv_python(repo: Path) -> Path:
    if os.name == "nt":
        return repo / ".venv_clean" / "Scripts" / "python.exe"
    return repo / ".venv_clean" / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--quick", action="store_true", help="Run bounded clean-room checks.")
    args = parser.parse_args()
    if not args.quick:
        parser.error("Only --quick mode is currently implemented.")

    repo = copy_repo(args.work_dir)
    runner = Runner(repo)
    runner.log(f"Copied repository from {ROOT} to {repo}")

    try:
        runner.log("1. Create fresh virtual environment")
        runner.run([sys.executable, "-m", "venv", ".venv_clean"])
        py = str(venv_python(repo))

        runner.log("2. Install package from clean copy")
        runner.run([py, "-m", "pip", "install", "-U", "pip"])
        runner.run([py, "-m", "pip", "install", "-e", ".[experiments,validation,dev]"])

        runner.log("3. Run bounded reproducibility check")
        runner.run([py, "scripts/run_clean_repro_check.py", "--quick"])

        runner.log("RESULT true clean-room check passed")
        return 0
    except Exception as exc:
        runner.log(f"RESULT true clean-room check failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
