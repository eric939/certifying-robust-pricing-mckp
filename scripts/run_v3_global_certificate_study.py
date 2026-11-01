#!/usr/bin/env python3
"""Build independently checked exact global certificates for Paper-A rows.

The default input is the frozen 45-row exact-comparison release.  Instances are
reconstructed from the recorded seed/design, solved again with the public exact
solver, and certified by :mod:`robust_mckp.global_certificate`.  The dated
release is read-only; this script always creates a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments_nested"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments_nested._common import (  # noqa: E402
    build_prefix_instance,
    compute_hulls_for_theta,
    extract_arrays,
    make_master_portfolio,
)
from robust_mckp import (  # noqa: E402
    GlobalThetaBNBConfig,
    PricingInstance,
    solve_reference as solve,
    solve_global_theta_bnb,
)
from robust_mckp.certificate import certificate_is_feasible  # noqa: E402
from robust_mckp.greedy import greedy_lp  # noqa: E402
from robust_mckp.global_certificate import (  # noqa: E402
    GlobalDualCertificate,
    build_global_certificate,
    original_theta_candidates,
    verify_global_certificate,
)


DEFAULT_RELEASE = (
    ROOT / "results" / "release" / "2026-08-09-paper-a-final-r2"
)
DEFAULT_OUTPUT = ROOT / "results" / "local" / "v3-global-certificate-study"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return float("nan")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return float("nan")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def _instance(seed: int, n: int, m: int, gamma: int):
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


def _exact_fixed_theta_capacity(
    instance: PricingInstance, theta: Fraction
) -> Fraction:
    stars = []
    for group in instance.items:
        transformed = [
            Fraction.from_float(float(option.margin))
            - max(
                Fraction(0),
                abs(Fraction.from_float(float(option.uncertainty))) - theta,
            )
            for option in group
        ]
        stars.append(max(transformed))
    return sum(stars, Fraction(0)) - int(instance.gamma) * theta


def _selection_objective(
    instance: PricingInstance, selections: Sequence[int]
) -> float:
    return math.fsum(
        float(instance.items[item][int(option)].value)
        for item, option in enumerate(selections)
    )


def _best_formal_round_down(instance: PricingInstance) -> List[int]:
    """Reconstruct the best literal lower-endpoint LP rounding over all theta."""

    values, margins, uncertainties = extract_arrays(instance)
    best_value = float("-inf")
    best_selection: List[int] | None = None
    for exact_theta in original_theta_candidates(instance):
        exact_capacity = _exact_fixed_theta_capacity(instance, exact_theta)
        if exact_capacity < 0:
            continue
        theta = float(exact_theta)
        hulls, _s_star_sum = compute_hulls_for_theta(
            values, margins, uncertainties, theta
        )
        lp = greedy_lp(hulls, float(exact_capacity))
        vertices = []
        for position in lp.positions:
            if position.lower_vertex == position.upper_vertex:
                vertices.append(position.lower_vertex)
            elif position.lambda_ >= 1.0:
                vertices.append(position.upper_vertex)
            else:
                vertices.append(position.lower_vertex)
        selections = [
            int(hulls[item].option_indices[vertex])
            for item, vertex in enumerate(vertices)
        ]
        if not certificate_is_feasible(instance, selections):
            raise RuntimeError(
                "formal round-down failed its exact sorted-Gamma check at "
                f"theta={theta.hex()}"
            )
        objective = _selection_objective(instance, selections)
        if objective > best_value:
            best_value = objective
            best_selection = selections
    if best_selection is None:
        raise RuntimeError("no exact-feasible formal round-down selection exists")
    return best_selection


def _optimal_multiplier_map(
    certificate: GlobalDualCertificate,
) -> Dict[Fraction, Fraction]:
    return {
        record.theta: record.multiplier
        for record in certificate.thresholds
        if record.feasible and record.multiplier is not None
    }


def _certificate_columns(
    label: str,
    certificate: GlobalDualCertificate,
    certificate_path: Path,
    output: Path,
) -> Dict[str, Any]:
    return {
        f"{label}_H": str(certificate.h),
        f"{label}_H_float": float(certificate.h),
        f"{label}_absolute_gap": str(certificate.absolute_gap),
        f"{label}_absolute_gap_float": float(certificate.absolute_gap),
        f"{label}_relative_width": (
            "" if certificate.relative_width is None else str(certificate.relative_width)
        ),
        f"{label}_relative_width_float": (
            float("nan")
            if certificate.relative_width is None
            else float(certificate.relative_width)
        ),
        f"{label}_robust_residual": str(certificate.robust_residual),
        f"{label}_robust_residual_float": float(certificate.robust_residual),
        f"{label}_certificate_file": str(certificate_path.relative_to(output)),
        f"{label}_certificate_sha256": _sha256(certificate_path),
    }


def _write_manifest(directory: Path) -> None:
    entries = []
    for path in sorted(
        candidate
        for candidate in directory.rglob("*")
        if candidate.is_file() and candidate.name != "MANIFEST.sha256"
    ):
        entries.append(f"{_sha256(path)}  {path.relative_to(directory)}")
    (directory / "MANIFEST.sha256").write_text(
        "\n".join(entries) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    release = args.release_dir.resolve()
    exact_csv = release / "exact_comparisons.csv"
    if not exact_csv.is_file():
        raise FileNotFoundError(f"missing frozen exact-comparison file: {exact_csv}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    certificate_dir = output / "certificates"
    certificate_dir.mkdir()
    input_dir = output / "inputs"
    input_dir.mkdir()
    archived_input = input_dir / "exact_comparisons.csv"
    shutil.copy2(exact_csv, archived_input)

    with exact_csv.open(newline="", encoding="utf-8") as handle:
        frozen_rows = list(csv.DictReader(handle))
    if args.limit is not None:
        frozen_rows = frozen_rows[: args.limit]

    bnb_tolerance = 1e-9
    rows: List[Dict[str, Any]] = []
    for position, frozen in enumerate(frozen_rows, start=1):
        seed = int(frozen["seed"])
        n = int(frozen["n"])
        m = int(frozen["m"])
        gamma = int(frozen["gamma"])
        instance = _instance(seed, n, m, gamma)
        exact = solve_global_theta_bnb(
            instance,
            GlobalThetaBNBConfig(
                tolerance=bnb_tolerance,
                use_hullround_incumbent=True,
                use_fast_residual_lp_bound=True,
            ),
        )
        if exact.status != "optimal" or exact.selected_options is None:
            raise RuntimeError(
                f"exact solver failed for seed={seed}, n={n}, m={m}, gamma={gamma}: "
                f"{exact.status}"
            )

        formal_selection = _best_formal_round_down(instance)
        completed = solve(instance, upgrade_completion=True)
        if not completed.is_feasible:
            raise RuntimeError(f"completion failed for row {position}")

        exact_certificate = build_global_certificate(
            instance, exact.selected_options
        )
        multipliers = _optimal_multiplier_map(exact_certificate)
        certificates = {
            "formal": build_global_certificate(
                instance, formal_selection, multipliers=multipliers
            ),
            "completed": build_global_certificate(
                instance, completed.selections, multipliers=multipliers
            ),
            "exact_bnb": exact_certificate,
        }
        if not (
            certificates["formal"].h
            <= certificates["completed"].h
            <= certificates["exact_bnb"].h
        ):
            raise RuntimeError(
                f"incumbent ordering failed for row {position}: "
                "expected formal <= completed <= exact_bnb"
            )
        if len({certificate.u for certificate in certificates.values()}) != 1:
            raise AssertionError("incumbent certificates disagree on sharp global U")

        certificate_paths: Dict[str, Path] = {}
        certificate_metrics: Dict[str, Dict[str, float | int]] = {}
        for label, certificate in certificates.items():
            raw = certificate.to_json(indent=2) + "\n"
            verify_start = time.perf_counter()
            verification = verify_global_certificate(instance, raw)
            verify_seconds = time.perf_counter() - verify_start
            if not verification.valid:
                raise RuntimeError(
                    f"{label} certificate verification failed for row {position}: "
                    + "; ".join(verification.errors)
                )
            certificate_name = (
                f"seed-{seed}_n-{n}_m-{m}_gamma-{gamma}_{label}.json"
            )
            certificate_path = certificate_dir / certificate_name
            certificate_path.write_text(raw, encoding="utf-8")
            certificate_paths[label] = certificate_path
            certificate_metrics[label] = {
                "bytes": len(raw.encode("utf-8")),
                "verify_seconds": verify_seconds,
            }

        frozen_objective = float(frozen["bnb_objective"])
        objective_difference = abs(
            float(certificates["exact_bnb"].h) - frozen_objective
        )
        if objective_difference > 1e-6:
            raise RuntimeError(
                f"reconstructed objective differs from frozen row {position}: "
                f"{objective_difference}"
            )

        reference_certificate = certificates["exact_bnb"]
        feasible_thresholds = sum(
            record.feasible for record in reference_certificate.thresholds
        )
        row: Dict[str, Any] = {
            "seed": seed,
            "n": n,
            "m": m,
            "gamma": gamma,
            "instance_sha256": reference_certificate.instance_hash,
            "threshold_count": len(reference_certificate.thresholds),
            "feasible_threshold_count": feasible_thresholds,
            "U": str(reference_certificate.u),
            "U_float": float(reference_certificate.u),
            "frozen_bnb_objective": frozen_objective,
            "frozen_objective_abs_difference": objective_difference,
            "bnb_reported_status": exact.status,
            "bnb_tolerance": bnb_tolerance,
            "bnb_lower_bound": exact.lower_bound,
            "bnb_upper_bound": exact.upper_bound,
            "bnb_absolute_gap": exact.absolute_gap,
            "formal_verified": True,
            "completed_verified": True,
            "exact_bnb_verified": True,
        }
        for label, certificate in certificates.items():
            row.update(
                _certificate_columns(
                    label,
                    certificate,
                    certificate_paths[label],
                    output,
                )
            )
            row[f"{label}_certificate_bytes"] = certificate_metrics[label]["bytes"]
            row[f"{label}_verify_seconds"] = certificate_metrics[label][
                "verify_seconds"
            ]
        rows.append(row)

    _write_csv(output / "global_certificate_study.csv", rows)
    widths = {
        label: [
            float(row[f"{label}_relative_width_float"])
            for row in rows
            if math.isfinite(float(row[f"{label}_relative_width_float"]))
        ]
        for label in ("formal", "completed", "exact_bnb")
    }
    certificate_bytes = [
        float(row[f"{label}_certificate_bytes"])
        for row in rows
        for label in ("formal", "completed", "exact_bnb")
    ]
    verification_seconds = [
        float(row[f"{label}_verify_seconds"])
        for row in rows
        for label in ("formal", "completed", "exact_bnb")
    ]
    summary = {
        "schema_version": 1,
        "purpose": "Paper-A v3 exact global dual-certificate study",
        "source_rows": len(frozen_rows),
        "certificates_built": 3 * len(rows),
        "certificates_verified": sum(
            bool(row[f"{label}_verified"])
            for row in rows
            for label in ("formal", "completed", "exact_bnb")
        ),
        "max_frozen_objective_abs_difference": max(
            (float(row["frozen_objective_abs_difference"]) for row in rows),
            default=float("nan"),
        ),
        "relative_width_denominator": "U when U>0; undefined otherwise",
        "formal_median_relative_width": _median(widths["formal"]),
        "formal_p95_relative_width": _percentile(widths["formal"], 0.95),
        "formal_max_relative_width": max(widths["formal"], default=float("nan")),
        "completed_median_relative_width": _median(widths["completed"]),
        "completed_p95_relative_width": _percentile(widths["completed"], 0.95),
        "completed_max_relative_width": max(
            widths["completed"], default=float("nan")
        ),
        "exact_bnb_median_relative_width": _median(widths["exact_bnb"]),
        "exact_bnb_p95_relative_width": _percentile(widths["exact_bnb"], 0.95),
        "exact_bnb_max_relative_width": max(
            widths["exact_bnb"], default=float("nan")
        ),
        "median_threshold_count": _median(
            float(row["threshold_count"]) for row in rows
        ),
        "median_certificate_bytes": _median(certificate_bytes),
        "max_certificate_bytes": max(certificate_bytes, default=float("nan")),
        "median_verification_seconds": _median(verification_seconds),
        "p95_verification_seconds": _percentile(verification_seconds, 0.95),
        "max_verification_seconds": max(
            verification_seconds, default=float("nan")
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_files = [
        ROOT / "src" / "robust_mckp" / "global_certificate.py",
        ROOT / "src" / "robust_mckp" / "certificate.py",
        ROOT / "src" / "robust_mckp" / "exact_bnb.py",
        ROOT / "src" / "robust_mckp" / "greedy.py",
        ROOT / "src" / "robust_mckp" / "hull.py",
        ROOT / "src" / "robust_mckp" / "rounding.py",
        ROOT / "src" / "robust_mckp" / "solver.py",
        ROOT / "experiments_nested" / "_common.py",
        Path(__file__).resolve(),
    ]
    protocol = {
        "schema_version": 1,
        "input_release_logical_name": "2026-08-09-paper-a-final-r2",
        "input_table": str(archived_input.relative_to(output)),
        "input_exact_comparisons_sha256": _sha256(archived_input),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "coefficient_interpretation": "Fraction.from_float(binary64)",
        "theta_policy": "zero plus every original binary64-distinct absolute deviation",
        "multiplier_policy": "exact minimizer of each fixed-threshold one-dimensional LP dual",
        "multiplier_verification": "exact left/right subgradient optimality at every feasible threshold",
        "checker_policy": "recompute transformed data, multiplier optimality, and every U_theta from original instance coefficients",
        "incumbent_tracks": [
            "best literal lower-endpoint round-down over the full threshold family",
            "best rounding/repair path plus feasibility-preserving completion over the full threshold family",
            "tolerance-controlled tailored B&B incumbent",
        ],
        "bnb_tolerance": bnb_tolerance,
        "bnb_status_interpretation": (
            "reported optimal status closes floating LP bounds within tolerance; "
            "it is not an exact-rational integer-optimality proof"
        ),
        "source_file_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_files
        },
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_manifest(output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional leading-row limit for a smoke study.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
