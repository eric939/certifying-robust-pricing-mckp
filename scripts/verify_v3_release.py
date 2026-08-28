#!/usr/bin/env python3
"""Fail-closed verification of the manuscript-free Paper A v3 release."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments_nested"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments_nested._common import build_prefix_instance, make_master_portfolio  # noqa: E402
from robust_mckp import Option, PricingInstance  # noqa: E402
from robust_mckp.global_certificate import verify_global_certificate  # noqa: E402


DEFAULT_RELEASE = ROOT / "results" / "release" / "2026-08-28-paper-a-v3"
FORBIDDEN_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".docx",
    ".fdb_latexmk",
    ".fls",
    ".latex",
    ".pdf",
    ".synctex",
    ".tex",
}
FORBIDDEN_PARTS = {"build", "manuscript", "papers", "submission_packages"}
FORBIDDEN_RAW_NAMES = {"upccer.csv", "wcer.zip"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"V3 RELEASE VERIFY: FAIL: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        if relative.startswith("./"):
            relative = relative[2:]
        require(len(digest) == 64, f"invalid manifest digest for {relative}")
        require(relative not in entries, f"duplicate manifest entry {relative}")
        entries[relative] = digest
    return entries


def verify_manifest(root: Path) -> int:
    manifest_path = root / "MANIFEST.sha256"
    require(manifest_path.is_file(), f"missing manifest in {root}")
    manifest = read_manifest(manifest_path)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    require(set(manifest) == actual, f"manifest does not cover every file in {root}")
    for relative, expected in manifest.items():
        require(sha256(root / relative) == expected, f"hash mismatch: {relative}")
    return len(manifest)


def controlled_instance(seed: int, n: int, m: int, gamma: int) -> PricingInstance:
    master = make_master_portfolio(
        seed=seed,
        n_max=n,
        m_max=m,
        min_admissible_menu=min(8, m),
    )
    return build_prefix_instance(
        master,
        n=n,
        m=m,
        alpha=0.10,
        gamma=gamma,
    ).instance


def verify_controlled_study(release: Path) -> int:
    study = release / "global-certificate-study"
    verify_manifest(study)
    with (study / "global_certificate_study.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((study / "summary.json").read_text(encoding="utf-8"))
    require(len(rows) == 45, "controlled study must contain 45 rows")
    require(int(summary["source_rows"]) == 45, "controlled summary row mismatch")
    require(int(summary["certificates_built"]) == 135, "controlled certificate count mismatch")
    require(int(summary["certificates_verified"]) == 135, "a controlled certificate was not verified")

    checked = 0
    for row in rows:
        instance = controlled_instance(
            int(row["seed"]), int(row["n"]), int(row["m"]), int(row["gamma"])
        )
        for label in ("formal", "completed", "exact_bnb"):
            relative = row[f"{label}_certificate_file"]
            raw = (study / relative).read_text(encoding="utf-8")
            require(
                sha256(study / relative) == row[f"{label}_certificate_sha256"],
                f"certificate hash mismatch: {relative}",
            )
            result = verify_global_certificate(instance, raw)
            require(result.valid, f"invalid {relative}: {'; '.join(result.errors)}")
            checked += 1
    require(checked == 135, "not every controlled certificate was checked")
    return checked


def pilot_instance(payload: dict[str, object], target: dict[str, object]) -> PricingInstance:
    calibration = payload["calibration"]
    model = payload["model"]
    assert isinstance(calibration, dict) and isinstance(model, dict)
    radius = float(calibration["radius"])
    gamma = int(calibration["gamma"])
    kappa = float(target["kappa"])
    downward_factor = radius
    upward_factor = radius / (1.0 - radius) if radius > 0.0 else 0.0
    groups: list[list[Option]] = []
    items = model["items"]
    assert isinstance(items, list)
    for item in items:
        assert isinstance(item, dict)
        cost = float(item["accounting_unit_cost"])
        menu = item["menu"]
        demand = item["demand"]
        assert isinstance(menu, list) and isinstance(demand, list)
        group: list[Option] = []
        for raw_price, raw_demand in zip(menu, demand, strict=True):
            price = float(raw_price)
            quantity = float(raw_demand)
            unit_surplus = price - kappa * cost
            adverse_factor = (
                downward_factor if unit_surplus >= 0.0 else upward_factor
            )
            group.append(
                Option(
                    value=float(price * quantity),
                    margin=float(unit_surplus * quantity),
                    uncertainty=float(abs(unit_surplus) * adverse_factor * quantity),
                    price=price,
                )
            )
        groups.append(group)
    return PricingInstance(items=groups, gamma=gamma, name="dominicks-cereal-v3-pilot")


def verify_pilot(release: Path) -> int:
    pilot = release / "dominicks-pilot"
    payload = json.loads(
        (pilot / "dominicks_cereal_pilot.json").read_text(encoding="utf-8")
    )
    require(
        payload["status"] == "failed_availability_and_unmodeled_dependence_gate",
        "the released negative pilot gate changed",
    )
    calibration = payload["calibration"]
    require(not calibration["availability_gate_passed"], "availability gate unexpectedly passes")
    require(not calibration["reliability_gate_passed"], "reliability gate unexpectedly passes")
    require(int(calibration["test_complete_weeks"]) == 7, "pilot complete-week count changed")
    require(int(calibration["test_calendar_weeks"]) == 79, "pilot calendar-week count changed")

    checked = 0
    for target in payload["targets"].values():
        if not target["robust_feasible"]:
            continue
        metadata = target["robust"]["global_certificate"]
        path = pilot / "certificates" / metadata["file"]
        require(sha256(path) == metadata["sha256"], f"pilot certificate hash mismatch: {path.name}")
        result = verify_global_certificate(
            pilot_instance(payload, target), path.read_text(encoding="utf-8")
        )
        require(result.valid, f"invalid {path.name}: {'; '.join(result.errors)}")
        checked += 1
    require(checked == 3, "pilot must contain three feasible exact certificates")
    return checked


def verify_public_tree() -> None:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    for relative in filter(None, tracked):
        path = Path(relative)
        require(not (set(path.parts) & FORBIDDEN_PARTS), f"tracked publication path: {relative}")
        suffixes = {suffix.lower() for suffix in path.suffixes}
        require(not (suffixes & FORBIDDEN_SUFFIXES), f"tracked manuscript/build file: {relative}")
        require(path.name.lower() not in FORBIDDEN_RAW_NAMES, f"tracked restricted input: {relative}")
        require("data/raw" not in path.as_posix().lower(), f"tracked raw-data path: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    require(release.is_dir(), f"missing release directory {release}")
    evidence_files = verify_manifest(release)
    controlled = verify_controlled_study(release)
    pilot = verify_pilot(release)
    verify_public_tree()
    print(
        "V3 RELEASE VERIFY: PASS "
        f"({evidence_files} evidence files, {controlled} controlled certificates, "
        f"{pilot} pilot certificates)"
    )


if __name__ == "__main__":
    main()
