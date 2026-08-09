#!/usr/bin/env python3
"""Fail-closed verification of the released code and computational evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(release: Path) -> None:
    manifest = release / "MANIFEST.sha256"
    if not manifest.is_file():
        raise RuntimeError("release manifest is missing")
    recorded: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        recorded.add(relative)
        path = release / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"manifest mismatch: {relative}")
    actual = {
        str(path.relative_to(release))
        for path in release.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    if recorded != actual:
        raise RuntimeError("manifest does not cover every released evidence file")


def verify_science(release: Path) -> None:
    summary = json.loads((release / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (release / "mathematical_audit.json").read_text(encoding="utf-8")
    )
    if not audit.get("passed") or int(audit.get("failure_count", 1)) != 0:
        raise RuntimeError("mathematical audit did not pass")
    exact = int(summary["exact_comparison_rows"])
    if int(summary["exact_bnb_optimal_rows"]) != exact:
        raise RuntimeError("an exact branch-and-bound row was not certified")
    if int(summary["highs_certified_rows"]) != exact:
        raise RuntimeError("an independent MILP comparison was not certified")
    cert = int(summary["certificate_rows"])
    if int(summary["certificate_feasible_rows"]) != cert:
        raise RuntimeError("a returned decision failed exact robust feasibility")
    if int(summary["rounding_bound_rows"]) != cert:
        raise RuntimeError("a one-item rounding bound failed")
    scale = int(summary["scalability_rows"])
    if int(summary["scalability_feasible_rows"]) != scale:
        raise RuntimeError("a scalability row failed robust certification")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    verify_manifest(release)
    verify_science(release)
    print("Paper A code and evidence verification passed")


if __name__ == "__main__":
    main()
