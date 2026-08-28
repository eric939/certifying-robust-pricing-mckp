#!/usr/bin/env python3
"""Freeze exact binary64 inputs for portable verification of v3 certificates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments_nested", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robust_mckp.global_certificate import instance_sha256  # noqa: E402
from run_v3_global_certificate_study import _instance  # noqa: E402


DEFAULT_STUDY = (
    ROOT
    / "results"
    / "release"
    / "2026-08-28-paper-a-v3"
    / "global-certificate-study"
    / "global_certificate_study.csv"
)
DEFAULT_OUTPUT = ROOT / "results" / "local" / "v3-controlled-instances"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_instance(instance) -> dict[str, object]:
    return {
        "schema_version": 1,
        "coefficient_encoding": "python_float_hex_binary64",
        "gamma": int(instance.gamma),
        "name": instance.name,
        "items": [
            [
                {
                    "value": float(option.value).hex(),
                    "margin": float(option.margin).hex(),
                    "uncertainty": float(option.uncertainty).hex(),
                    "price": (
                        None
                        if option.price is None
                        else float(option.price).hex()
                    ),
                }
                for option in group
            ]
            for group in instance.items
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-csv", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_csv = args.study_csv.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    with study_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 45:
        raise ValueError(f"expected 45 study rows, found {len(rows)}")

    records = []
    for row in rows:
        seed = int(row["seed"])
        n = int(row["n"])
        m = int(row["m"])
        gamma = int(row["gamma"])
        instance = _instance(seed, n, m, gamma)
        observed_hash = instance_sha256(instance)
        if observed_hash != row["instance_sha256"]:
            raise ValueError(
                f"reconstructed instance hash mismatch for seed={seed}, n={n}, "
                f"m={m}, gamma={gamma}"
            )
        filename = f"seed-{seed}_n-{n}_m-{m}_gamma-{gamma}.json"
        path = output / filename
        path.write_text(
            json.dumps(encode_instance(instance), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records.append(
            {
                "file": filename,
                "file_sha256": sha256(path),
                "instance_sha256": observed_hash,
                "seed": seed,
                "n": n,
                "m": m,
                "gamma": gamma,
            }
        )

    index = {
        "schema_version": 1,
        "purpose": "cross-platform exact inputs for Paper A v3 certificate verification",
        "coefficient_encoding": "python_float_hex_binary64",
        "source_study_csv_sha256": sha256(study_csv),
        "instances": records,
    }
    (output / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} exact instance snapshots to {output}")


if __name__ == "__main__":
    main()
