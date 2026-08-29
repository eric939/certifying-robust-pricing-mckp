#!/usr/bin/env python3
"""Fail-closed verification of the Paper-A v3 independent-r/q campaign."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robust_mckp.global_certificate import GlobalDualCertificate  # noqa: E402
from robust_mckp.global_certificate import verify_global_certificate  # noqa: E402
from scripts.run_v3_independent_rq_campaign import (  # noqa: E402
    CALIBRATION_CELLS,
    CAMPAIGN_ID,
    floating_full_family_interval,
    load_instance_payload,
    summarize_campaign,
    write_macros,
)


DEFAULT_RELEASE = ROOT / "results" / "release" / CAMPAIGN_ID
INTEGER_TEXT = re.compile(r"[+-]?\d+")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("V3 INDEPENDENT-RQ VERIFY: FAIL: " + message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), "missing table: %s" % path.name)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    require(value in {"true", "false"}, "noncanonical boolean: %r" % value)
    return value == "true"


def typed_row(row: Mapping[str, str]) -> dict[str, Any]:
    """Convert canonical campaign CSV text back to generator value types."""

    typed: dict[str, Any] = {}
    for field, value in row.items():
        if value == "":
            typed[field] = None
        elif value in {"true", "false"}:
            typed[field] = value == "true"
        elif INTEGER_TEXT.fullmatch(value):
            typed[field] = int(value)
        else:
            try:
                typed[field] = float(value)
            except ValueError:
                typed[field] = value
    return typed


def finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        require(False, "%s is not numeric" % label)
        raise AssertionError("unreachable")
    require(math.isfinite(number), "%s is not finite" % label)
    return number


def require_close(
    actual: Any,
    expected: Any,
    label: str,
    *,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-10,
) -> None:
    actual_number = finite_float(actual, label)
    expected_number = finite_float(expected, label + " expected")
    require(
        math.isclose(
            actual_number,
            expected_number,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ),
        "%s mismatch: stored=%r recomputed=%r"
        % (label, actual_number, expected_number),
    )


def verify_pricing_payload(payload: Mapping[str, Any]) -> None:
    """Reconstruct the primitive-to-option pricing identities in a snapshot."""

    require(int(payload.get("schema_version", -1)) == 1, "unsupported instance schema")
    require(
        payload.get("coefficient_interpretation") == "exact binary64 hexadecimal",
        "unexpected coefficient interpretation",
    )
    n = int(payload["n"])
    m = int(payload["m"])
    reference = [float.fromhex(value) for value in payload["reference_prices"]]
    costs = [float.fromhex(value) for value in payload["unit_costs"]]
    weights = [float.fromhex(value) for value in payload["weights"]]
    tolerances = [float.fromhex(value) for value in payload["tolerances"]]
    groups = list(payload["groups"])
    kappa = float.fromhex(payload["kappa"])
    radius = float.fromhex(payload["radius"])

    require(n > 0 and m > 0, "invalid instance dimensions")
    require(
        all(len(values) == n for values in (reference, costs, weights, tolerances, groups)),
        "primitive vector length mismatch",
    )
    require(max(len(group) for group in groups) == m, "stored menu width mismatch")
    require(0.0 <= radius < 1.0, "radius is outside [0,1)")
    require(math.isfinite(kappa) and kappa > 0.0, "invalid margin target")
    require(
        all(math.isfinite(value) and value > 0.0 for value in reference),
        "invalid reference price",
    )
    require(
        all(math.isfinite(value) and value > 0.0 for value in costs),
        "invalid unit cost",
    )
    require(
        all(math.isfinite(value) and value > 0.0 for value in weights),
        "invalid exposure weight",
    )
    require(
        all(math.isfinite(value) and value >= 0.0 for value in tolerances),
        "invalid admissibility tolerance",
    )
    require(
        all(r != q for r, q in zip(reference, costs)),
        "reference prices and unit costs are not separated",
    )

    for i, group in enumerate(groups):
        require(group, "empty option group %d" % i)
        lower = (1.0 - tolerances[i]) * reference[i]
        upper = (1.0 + tolerances[i]) * reference[i]
        for j, raw_option in enumerate(group):
            label = "primitive option (%d,%d)" % (i, j)
            require(raw_option["price"] is not None, label + " has no price")
            price = float.fromhex(raw_option["price"])
            value = float.fromhex(raw_option["value"])
            margin = float.fromhex(raw_option["margin"])
            uncertainty = float.fromhex(raw_option["uncertainty"])
            require(
                all(math.isfinite(number) for number in (price, value, margin, uncertainty)),
                label + " contains a nonfinite coefficient",
            )
            require(price > 0.0 and value > 0.0, label + " has a nonpositive price/value")
            require(uncertainty >= 0.0, label + " has a negative endpoint loss")
            admission_tol = 1e-12 * max(1.0, abs(lower), abs(upper), abs(price))
            require(
                lower - admission_tol <= price <= upper + admission_tol,
                label + " violates reference-price admissibility",
            )

            demand = value / (weights[i] * price)
            unit_surplus = price - kappa * costs[i]
            expected_margin = weights[i] * unit_surplus * demand
            adverse_factor = radius if unit_surplus >= 0.0 else radius / (1.0 - radius)
            expected_uncertainty = (
                weights[i] * abs(unit_surplus) * adverse_factor * demand
            )
            require_close(
                margin,
                expected_margin,
                label + " margin from (q,price,value)",
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            require_close(
                uncertainty,
                expected_uncertainty,
                label + " endpoint loss from radius",
                rel_tol=1e-12,
                abs_tol=1e-9,
            )


def verify_manifest(release: Path) -> int:
    path = release / "manifest.json"
    require(path.is_file(), "missing manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload["campaign_id"] == CAMPAIGN_ID, "manifest campaign ID mismatch")
    records = payload["files"]
    require(int(payload["file_count"]) == len(records), "manifest count mismatch")
    indexed = {record["path"]: record for record in records}
    require(len(indexed) == len(records), "duplicate manifest path")
    actual = {
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    require(set(indexed) == actual, "manifest does not cover exactly the release")
    for relative, record in indexed.items():
        artifact = release / relative
        require(artifact.stat().st_size == int(record["bytes"]), "size mismatch: " + relative)
        require(sha256(artifact) == record["sha256"], "hash mismatch: " + relative)
    return len(records)


def verify_exact_table(release: Path, table_name: str) -> list[dict[str, Any]]:
    """Verify exact artifacts and recompute every certificate-derived CSV field."""

    rows = read_csv(release / table_name)
    typed_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        label = "%s row %d" % (table_name, row_index + 1)
        instance_path = release / row["instance_file"]
        certificate_path = release / row["certificate_file"]
        require(instance_path.is_file(), "missing instance: " + row["instance_file"])
        require(certificate_path.is_file(), "missing certificate: " + row["certificate_file"])
        require(sha256(instance_path) == row["instance_sha256"], "instance row hash mismatch")
        require(sha256(certificate_path) == row["certificate_sha256"], "certificate row hash mismatch")
        payload = json.loads(instance_path.read_text(encoding="utf-8"))
        verify_pricing_payload(payload)
        instance = load_instance_payload(payload)
        margins = [option.margin for group in instance.items for option in group]
        require(any(value < 0.0 for value in margins), "instance lacks negative-surplus options")
        require(any(value > 0.0 for value in margins), "instance lacks positive-surplus options")
        raw_certificate = certificate_path.read_text(encoding="utf-8")
        certificate = GlobalDualCertificate.from_json(raw_certificate)
        result = verify_global_certificate(instance, raw_certificate)
        require(result.valid, "invalid exact certificate: " + "; ".join(result.errors))
        require(certificate.schema_version == 2, "certificate is not schema v2")
        require(result.rounding_provenance_verified, "rounding provenance was not verified")
        require(certificate.rounding_provenance is not None, "missing rounding provenance")
        require(
            certificate.absolute_gap <= certificate.rounding_provenance.rounding_loss,
            "U-H exceeds literal rounding loss",
        )
        provenance = certificate.rounding_provenance

        require(int(row["seed"]) == int(payload["seed"]), label + " seed mismatch")
        require(int(row["n"]) == int(payload["n"]), label + " n mismatch")
        require(int(row["m"]) == int(payload["m"]), label + " m mismatch")
        require(int(row["gamma"]) == int(payload["gamma"]), label + " Gamma mismatch")
        require_close(
            row["radius"],
            float.fromhex(payload["radius"]),
            label + " radius",
            abs_tol=0.0,
        )
        require(len(raw_certificate.encode("utf-8")) == int(row.get("certificate_bytes", len(raw_certificate.encode("utf-8")))), label + " certificate byte count mismatch")
        require(parse_bool(row["certificate_verified"]), "row says certificate unverified")
        require(parse_bool(row["rounding_provenance_verified"]), "row says provenance unverified")
        require(parse_bool(row["exact_gap_leq_rounding_loss"]), "row says gap inequality failed")
        require_close(
            row["exact_rounding_loss"],
            float(provenance.rounding_loss),
            label + " exact rounding loss",
        )

        if table_name == "certificate-study.csv":
            require(row["regime"] == payload["regime"], label + " regime mismatch")
            require(
                row["gamma_mode"] == payload["gamma_mode"],
                label + " Gamma mode mismatch",
            )
            require(int(row["certificate_schema_version"]) == 2, label + " schema mismatch")
            negative = sum(value < 0.0 for value in margins)
            positive = sum(value > 0.0 for value in margins)
            require(
                int(row["negative_margin_options"]) == negative,
                label + " negative-option count mismatch",
            )
            require(
                int(row["positive_margin_options"]) == positive,
                label + " positive-option count mismatch",
            )
            require(
                int(row["threshold_count"]) == len(certificate.thresholds),
                label + " threshold count mismatch",
            )
            require(
                int(row["feasible_threshold_count"])
                == sum(record.feasible for record in certificate.thresholds),
                label + " feasible-threshold count mismatch",
            )
            require_close(row["completed_H_exact"], float(certificate.h), label + " H")
            require_close(row["U_exact"], float(certificate.u), label + " U")
            require_close(
                row["completed_relative_width_exact"],
                float(certificate.relative_width),
                label + " relative width",
                abs_tol=1e-15,
            )
            require_close(
                row["robust_residual_exact"],
                float(certificate.robust_residual),
                label + " robust residual",
            )
            require_close(
                row["exact_adjacent_value_jump"],
                float(provenance.adjacent_value_jump),
                label + " adjacent value jump",
            )

            sweep = floating_full_family_interval(instance)
            require_close(
                row["literal_H_float"],
                sweep["literal_H_float"],
                label + " literal H",
            )
            require_close(
                row["literal_relative_width_float"],
                (float(certificate.u) - float(sweep["literal_H_float"]))
                / float(certificate.u),
                label + " literal relative width",
                abs_tol=1e-15,
            )
            require_close(
                row["rounding_loss_at_theta_U_float"],
                sweep["rounding_loss_at_theta_U"],
                label + " numerical rounding loss",
            )
            require_close(
                row["fractional_jump_at_theta_U_float"],
                sweep["fractional_jump_at_theta_U"],
                label + " numerical adjacent jump",
            )

            require(row["highs_status"] == "OPTIMAL", label + " HiGHS status mismatch")
            require(parse_bool(row["highs_certified"]), label + " HiGHS check failed")
            highs_objective = finite_float(row["highs_objective"], label + " HiGHS objective")
            require(
                float(certificate.h) - 1e-7
                <= highs_objective
                <= float(certificate.u) + 1e-7,
                label + " HiGHS objective lies outside [H,U]",
            )
            require_close(
                row["completed_highs_absolute_difference"],
                abs(float(certificate.h) - highs_objective),
                label + " HiGHS/H difference",
            )
        elif table_name == "gamma-frontier.csv":
            require(payload["regime"] == "favorable", label + " frontier regime mismatch")
            require(
                payload["gamma_mode"] == "frontier_integer",
                label + " frontier mode mismatch",
            )
            require_close(row["H_exact"], float(certificate.h), label + " H")
            require_close(row["U_exact"], float(certificate.u), label + " U")
            require_close(
                row["absolute_gap_exact"],
                float(certificate.absolute_gap),
                label + " absolute gap",
            )
            require_close(
                row["relative_width_exact"],
                float(certificate.relative_width),
                label + " relative width",
                abs_tol=1e-15,
            )
            require_close(
                row["robust_residual_exact"],
                float(certificate.robust_residual),
                label + " robust residual",
            )
        else:
            require(False, "unsupported exact table: " + table_name)

        typed_rows.append(typed_row(row))
    return typed_rows


def verify_frontier_rows(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    expected_gamma = [int(value) for value in protocol["gamma_frontier"]["gamma_values"]]
    actual_gamma = [int(row["gamma"]) for row in rows]
    require(actual_gamma == expected_gamma, "Gamma-frontier values/order mismatch")
    require(rows and finite_float(rows[0]["H_exact"], "Gamma-zero H") > 0.0, "invalid Gamma-zero H")
    gamma_zero_h = float(rows[0]["H_exact"])
    previous_h = math.inf
    previous_u = math.inf
    for row in rows:
        gamma = int(row["gamma"])
        h = finite_float(row["H_exact"], "frontier H")
        u = finite_float(row["U_exact"], "frontier U")
        require(h <= previous_h + 1e-9, "frontier incumbent is not nonincreasing")
        require(u <= previous_u + 1e-9, "frontier U is not nonincreasing")
        previous_h, previous_u = h, u
        require_close(
            row["revenue_ratio_vs_gamma_zero"],
            h / gamma_zero_h,
            "frontier revenue ratio at Gamma=%d" % gamma,
        )
        require_close(
            row["revenue_cost_vs_gamma_zero"],
            1.0 - h / gamma_zero_h,
            "frontier revenue cost at Gamma=%d" % gamma,
        )


def verify_scaling_rows(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    design = [tuple(record) for record in protocol["scaling_design"]]
    observed = [
        (row["regime"], int(row["seed"]), int(row["n"]), int(row["m"]), row["gamma_mode"])
        for row in rows
    ]
    require(observed == design, "scaling design/order mismatch")
    for row in rows:
        label = "scaling %s/%s/n=%s" % (row["regime"], row["seed"], row["n"])
        n = int(row["n"])
        h = finite_float(row["completed_H_float"], label + " H")
        literal_h = finite_float(row["literal_H_float"], label + " literal H")
        u = finite_float(row["U_float"], label + " U")
        require(u > 0.0 and h <= u + 1e-7 and literal_h <= u + 1e-7, label + " invalid interval")
        completed_width = (u - h) / u
        literal_width = (u - literal_h) / u
        require_close(row["completed_relative_width_float"], completed_width, label + " completed width")
        require_close(row["literal_relative_width_float"], literal_width, label + " literal width")
        require_close(row["n_times_completed_relative_width"], n * completed_width, label + " n-times width")
        require_close(row["objective_per_item"], h / n, label + " objective per item")
        require(
            finite_float(row["rounding_loss_at_theta_U_float"], label + " rounding loss")
            <= finite_float(row["fractional_jump_at_theta_U_float"], label + " jump")
            + 1e-6,
            label + " rounding loss exceeds adjacent jump",
        )
        require(bool(row["certificate_feasible"]), label + " says incumbent infeasible")
        require(finite_float(row["robust_residual"], label + " robust residual") >= -1e-7, label + " negative robust residual")
        require(int(row["negative_margin_options"]) > 0, label + " lacks negative-surplus options")
        require(int(row["positive_margin_options"]) > 0, label + " lacks positive-surplus options")
        require(
            row["interpretation"] == "floating diagnostic; not an exact certificate",
            label + " interpretation mismatch",
        )


def verify_calibration_rows(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    calibration = protocol["calibration"]
    repetitions = int(calibration["repetitions_per_cell"])
    cell_roles = dict(CALIBRATION_CELLS)
    expected_order = [
        (cell, repetition)
        for cell, _ in CALIBRATION_CELLS
        for repetition in range(repetitions)
    ]
    observed_order = [(str(row["cell"]), int(row["repetition"])) for row in rows]
    require(observed_order == expected_order, "calibration cell/repetition order mismatch")
    for cell_index, (cell, _) in enumerate(CALIBRATION_CELLS):
        for row in (item for item in rows if item["cell"] == cell):
            repetition = int(row["repetition"])
            label = "calibration %s/%d" % (cell, repetition)
            require(row["role"] == cell_roles[cell], label + " role mismatch")
            require(int(row["seed"]) == 710000 + 10000 * cell_index + repetition, label + " seed mismatch")
            require(int(row["n"]) == int(calibration["n"]), label + " n mismatch")
            require(int(row["m"]) == int(calibration["m"]), label + " m mismatch")
            require(int(row["tune_size"]) == int(calibration["tuning_observations"]), label + " tuning size mismatch")
            require(int(row["calibration_size"]) == int(calibration["calibration_observations"]), label + " calibration size mismatch")
            require_close(row["alpha"], protocol["alpha"], label + " alpha", abs_tol=0.0)
            require(1 <= int(row["gamma"]) <= int(row["n"]), label + " Gamma outside range")
            radius = finite_float(row["radius"], label + " radius")
            require(0.0 <= radius < 1.0, label + " radius outside [0,1)")

            for policy in ("nominal", "robust", "box"):
                require(bool(row[policy + "_solution_feasible"]), label + " %s solution infeasible" % policy)
                objective = finite_float(row[policy + "_objective"], label + " %s objective" % policy)
                realized_margin = finite_float(row[policy + "_realized_margin"], label + " %s margin" % policy)
                expected_violation = realized_margin < -1e-8 * max(1.0, abs(objective))
                require(
                    bool(row[policy + "_violation"]) == expected_violation,
                    label + " %s violation flag mismatch" % policy,
                )
                require(
                    0 <= int(row[policy + "_negative_selected"]) <= int(row["n"]),
                    label + " %s negative-selection count invalid" % policy,
                )
                require(
                    finite_float(
                        row[policy + "_model_certificate_residual"],
                        label + " %s model residual" % policy,
                    )
                    >= -1e-7,
                    label + " %s model residual is negative" % policy,
                )

            nominal_objective = float(row["nominal_objective"])
            for policy in ("robust", "box"):
                require_close(
                    row[policy + "_revenue_ratio_vs_nominal"],
                    float(row[policy + "_objective"]) / nominal_objective,
                    label + " %s revenue ratio" % policy,
                )
            expected_covered = float(row["test_score"]) <= radius + 1e-12
            require(bool(row["set_covered"]) == expected_covered, label + " set-coverage flag mismatch")
            require(
                not (bool(row["set_covered"]) and bool(row["robust_violation"])),
                label + " violates coverage-to-feasibility containment",
            )


def require_equivalent(actual: Any, expected: Any, label: str) -> None:
    """Recursively compare a released JSON summary to its row-level rebuild."""

    if isinstance(expected, Mapping):
        require(isinstance(actual, Mapping), label + " is not an object")
        require(set(actual) == set(expected), label + " keys mismatch")
        for key in expected:
            require_equivalent(actual[key], expected[key], label + "." + str(key))
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        require(isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)), label + " is not an array")
        require(len(actual) == len(expected), label + " length mismatch")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            require_equivalent(actual_item, expected_item, "%s[%d]" % (label, index))
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(actual == expected, label + " mismatch")
        return
    if isinstance(expected, (int, float)):
        require_close(actual, expected, label, abs_tol=1e-12)
        return
    require(actual == expected, label + " mismatch")


def verify_summary_and_macros(
    release: Path,
    certificate_rows: Sequence[Mapping[str, Any]],
    scaling_rows: Sequence[Mapping[str, Any]],
    frontier_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rebuilt = summarize_campaign(
        certificate_rows, scaling_rows, frontier_rows, calibration_rows
    )
    stored = json.loads((release / "summary.json").read_text(encoding="utf-8"))
    require_equivalent(stored, rebuilt, "summary")
    with tempfile.TemporaryDirectory(prefix="v3-rq-macros-") as directory:
        expected_path = Path(directory) / "summary-macros.txt"
        write_macros(expected_path, rebuilt)
        expected_text = expected_path.read_text(encoding="utf-8")
    actual_text = (release / "summary-macros.txt").read_text(encoding="utf-8")
    require(actual_text == expected_text, "summary-macros.txt mismatch")
    return rebuilt


def verify_protocol(release: Path) -> dict[str, object]:
    protocol = json.loads((release / "protocol.json").read_text(encoding="utf-8"))
    require(protocol["campaign_id"] == CAMPAIGN_ID, "protocol campaign ID mismatch")
    for relative, expected in protocol["source_sha256"].items():
        require((ROOT / relative).is_file(), "missing protocol source: " + relative)
        require(sha256(ROOT / relative) == expected, "protocol source hash mismatch: " + relative)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    require(release.is_dir(), "missing release directory %s" % release)
    manifest_files = verify_manifest(release)
    protocol = verify_protocol(release)
    certificate_rows = verify_exact_table(release, "certificate-study.csv")
    frontier_rows = verify_exact_table(release, "gamma-frontier.csv")
    scaling_rows = [typed_row(row) for row in read_csv(release / "scaling-study.csv")]
    calibration_rows = [
        typed_row(row) for row in read_csv(release / "calibration-study.csv")
    ]
    verify_frontier_rows(frontier_rows, protocol)
    verify_scaling_rows(scaling_rows, protocol)
    verify_calibration_rows(calibration_rows, protocol)
    summary = verify_summary_and_macros(
        release,
        certificate_rows,
        scaling_rows,
        frontier_rows,
        calibration_rows,
    )
    certificate_count = len(certificate_rows)
    frontier_count = len(frontier_rows)
    require(
        certificate_count == len(protocol["certificate_design"]),
        "certificate design count mismatch",
    )
    require(
        len(scaling_rows) == len(protocol["scaling_design"]),
        "scaling design count mismatch",
    )
    expected_repetitions = int(protocol["calibration"]["repetitions_per_cell"])
    require(
        len(calibration_rows) == expected_repetitions * len(CALIBRATION_CELLS),
        "calibration design count mismatch",
    )
    expected_frontier = len(protocol["gamma_frontier"]["gamma_values"])
    require(frontier_count == expected_frontier, "Gamma-frontier count mismatch")
    require(
        int(summary["certificate_study"]["instances"]) == certificate_count,
        "certificate summary count mismatch",
    )
    require(
        int(summary["gamma_frontier"]["rows"]) == frontier_count,
        "frontier summary count mismatch",
    )
    require(
        int(summary["calibration_study"]["rows"]) == len(calibration_rows),
        "calibration summary count mismatch",
    )
    print(
        "V3 INDEPENDENT-RQ VERIFY: PASS "
        "(%d files, %d study certificates, %d frontier certificates, %d calibration rows)"
        % (manifest_files, certificate_count, frontier_count, len(calibration_rows))
    )


if __name__ == "__main__":
    main()
