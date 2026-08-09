#!/usr/bin/env python3
"""Fail-closed verification of the frozen Paper A evidence and manuscript."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "arxiv-v2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(release: Path) -> None:
    manifest = release / "MANIFEST.sha256"
    if not manifest.is_file():
        raise RuntimeError("release manifest is missing")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = release / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"manifest mismatch: {relative}")


def verify_science(release: Path) -> None:
    summary = json.loads((release / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads((release / "mathematical_audit.json").read_text(encoding="utf-8"))
    if not audit.get("passed") or int(audit.get("failure_count", 1)) != 0:
        raise RuntimeError("mathematical audit did not pass")
    exact = int(summary["exact_comparison_rows"])
    if int(summary["exact_bnb_optimal_rows"]) != exact or int(summary["highs_certified_rows"]) != exact:
        raise RuntimeError("not every exact-comparison row was certified")
    cert = int(summary["certificate_rows"])
    if int(summary["certificate_feasible_rows"]) != cert or int(summary["rounding_bound_rows"]) != cert:
        raise RuntimeError("feasibility or one-item certificate gate failed")
    scale = int(summary["scalability_rows"])
    if int(summary["scalability_feasible_rows"]) != scale:
        raise RuntimeError("a scalability row failed direct certification")


def verify_materialized_artifacts(release: Path) -> None:
    pairs = [
        (release / "generated" / "release_macros.tex", PAPER / "generated" / "release_macros.tex"),
        (release / "figures" / "release_evidence.pdf", PAPER / "figures" / "release_evidence.pdf"),
    ]
    for source, materialized in pairs:
        if not materialized.is_file() or digest(source) != digest(materialized):
            raise RuntimeError(f"paper artifact is stale: {materialized.relative_to(ROOT)}")


def compile_papers() -> None:
    tectonic = shutil.which("tectonic") or "/opt/homebrew/bin/tectonic"
    if not Path(tectonic).exists():
        raise RuntimeError("Tectonic is unavailable")
    for source in ("main.tex", "main_blind.tex"):
        subprocess.run([tectonic, "--keep-logs", source], cwd=PAPER, check=True, stdout=subprocess.DEVNULL)
    for log_name in ("main.log", "main_blind.log"):
        text = (PAPER / log_name).read_text(errors="ignore")
        forbidden = ["Overfull \\hbox", "undefined references", "undefined citation", "multiply defined", "missing file"]
        hits = [needle for needle in forbidden if needle.lower() in text.lower()]
        if hits:
            raise RuntimeError(f"{log_name} contains release-blocking warnings: {hits}")


def verify_identity() -> None:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise RuntimeError("pdftotext is unavailable")
    public_text = subprocess.check_output([pdftotext, str(PAPER / "main.pdf"), "-"], text=True)
    blind_text = subprocess.check_output([pdftotext, str(PAPER / "main_blind.pdf"), "-"], text=True)
    if "Zi Yuan Eric Shao" not in public_text or "github.com/eric939/certifying-robust-pricing-mckp" not in public_text:
        raise RuntimeError("public build is missing author or repository identity")
    patterns = [
        r"\bEric\b", r"\bShao\b", r"ershao", r"ETH Z(?:ü|u)rich",
        r"github\.com/eric939", r"2603\.18653", r"Kunoth", r"Mollet",
        r"Hannoversche", r"University of Cologne",
    ]
    hits = [pattern for pattern in patterns if re.search(pattern, blind_text, flags=re.I)]
    if hits:
        raise RuntimeError(f"blind PDF contains identity strings: {hits}")


def verify_lineage() -> None:
    current = ROOT / "paper" / "current"
    if not current.is_symlink() or current.resolve() != PAPER.resolve():
        raise RuntimeError("paper/current does not point to arxiv-v2")
    lineage = (ROOT / "PAPER_LINEAGE.md").read_text(encoding="utf-8")
    required = ["Paper A", "Paper B", "simultaneous", "arXiv:2603.18653"]
    if any(term not in lineage for term in required):
        raise RuntimeError("Paper A/Paper B lineage statement is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    verify_manifest(release)
    verify_science(release)
    verify_materialized_artifacts(release)
    verify_lineage()
    compile_papers()
    verify_identity()
    print("Paper A release verification passed")


if __name__ == "__main__":
    main()
