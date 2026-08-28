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
sys.path.insert(0, str(ROOT / "src"))
from robust_mckp import Option, PricingInstance
from robust_mckp.global_certificate import instance_sha256, verify_global_certificate


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


def load_instance(path: Path) -> PricingInstance:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload["schema_version"] == 1, f"unsupported instance schema: {path}")
    require(
        payload["coefficient_encoding"] == "python_float_hex_binary64",
        f"unsupported coefficient encoding: {path}",
    )
    groups = []
    for raw_group in payload["items"]:
        group = []
        for raw in raw_group:
            raw_price = raw["price"]
            group.append(
                Option(
                    value=float.fromhex(raw["value"]),
                    margin=float.fromhex(raw["margin"]),
                    uncertainty=float.fromhex(raw["uncertainty"]),
                    price=(None if raw_price is None else float.fromhex(raw_price)),
                )
            )
        groups.append(group)
    return PricingInstance(
        items=groups,
        gamma=int(payload["gamma"]),
        name=str(payload.get("name") or ""),
    )


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

    instance_dir = study / "instances"
    index = json.loads((instance_dir / "index.json").read_text(encoding="utf-8"))
    require(index["schema_version"] == 1, "unsupported controlled-instance index")
    require(
        index["source_study_csv_sha256"]
        == sha256(study / "global_certificate_study.csv"),
        "controlled-instance index is bound to a different study table",
    )
    require(len(index["instances"]) == 45, "controlled-instance index count mismatch")
    indexed = {record["file"]: record for record in index["instances"]}
    require(len(indexed) == 45, "duplicate controlled-instance index entry")

    checked = 0
    for row in rows:
        filename = (
            f"seed-{row['seed']}_n-{row['n']}_m-{row['m']}_gamma-{row['gamma']}.json"
        )
        require(filename in indexed, f"missing controlled instance index: {filename}")
        instance_path = instance_dir / filename
        record = indexed[filename]
        require(sha256(instance_path) == record["file_sha256"], f"instance file hash mismatch: {filename}")
        instance = load_instance(instance_path)
        observed_instance_hash = instance_sha256(instance)
        require(observed_instance_hash == record["instance_sha256"], f"instance index hash mismatch: {filename}")
        require(observed_instance_hash == row["instance_sha256"], f"study instance hash mismatch: {filename}")
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
