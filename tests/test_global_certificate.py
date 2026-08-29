from __future__ import annotations

import copy
import json
import random
from fractions import Fraction

import pytest

from robust_mckp import Option, PricingInstance
from robust_mckp.exact_bnb import brute_force_global_robust
from robust_mckp.global_certificate import (
    CERTIFICATE_SCHEMA_VERSION,
    GlobalDualCertificate,
    ROUNDING_PROVENANCE_SCHEMA_VERSION,
    build_global_certificate,
    instance_sha256,
    original_theta_candidates,
    verify_global_certificate,
)


def _random_tiny_instance(rng: random.Random) -> PricingInstance:
    n = rng.randint(1, 5)
    m = rng.randint(1, 4)
    items = []
    for _item in range(n):
        # The first option is a safe baseline, so every generated instance has
        # at least one robust-feasible selection for every Gamma.
        group = [Option(value=float(rng.randint(0, 20)), margin=8.0, uncertainty=0.0)]
        for _option in range(1, m):
            group.append(
                Option(
                    value=float(rng.randint(-5, 30)),
                    margin=float(rng.randint(-5, 12)),
                    uncertainty=float(rng.randint(-6, 6)),
                )
            )
        items.append(group)
    return PricingInstance(items=items, gamma=rng.randint(0, n))


def test_random_tiny_global_certificates_cover_exhaustive_optima() -> None:
    rng = random.Random(20260828)
    for _case in range(30):
        instance = _random_tiny_instance(rng)
        exhaustive = brute_force_global_robust(instance, tol=0.0)
        assert exhaustive.status == "optimal"
        assert exhaustive.selected_options is not None

        certificate = build_global_certificate(instance, exhaustive.selected_options)
        verification = verify_global_certificate(instance, certificate)

        assert verification.valid, verification.errors
        assert float(certificate.h) == exhaustive.objective_value
        assert certificate.u >= certificate.h
        assert certificate.absolute_gap == certificate.u - certificate.h
        if certificate.u > 0:
            assert certificate.relative_width == (
                certificate.u - certificate.h
            ) / certificate.u
        else:
            assert certificate.relative_width is None
        assert tuple(record.theta for record in certificate.thresholds) == (
            original_theta_candidates(instance)
        )


def test_close_binary64_breakpoints_are_both_retained() -> None:
    delta = 9e-11
    instance = PricingInstance(
        items=[
            [Option(value=1.0, margin=2.0, uncertainty=1.0)],
            [Option(value=1.0, margin=2.0, uncertainty=1.0 + delta)],
        ],
        gamma=1,
    )
    certificate = build_global_certificate(instance, [0, 0])
    candidates = original_theta_candidates(instance)

    assert len(candidates) == 3
    assert candidates[1] == Fraction.from_float(1.0)
    assert candidates[2] == Fraction.from_float(1.0 + delta)
    assert candidates[1] != candidates[2]
    assert verify_global_certificate(instance, certificate).valid


def test_checker_rejects_bound_and_threshold_tampering() -> None:
    instance = PricingInstance(
        items=[
            [Option(3.0, 2.0, 1.0), Option(5.0, 1.0, 0.5)],
            [Option(4.0, 2.0, 0.0), Option(6.0, 1.0, 0.25)],
        ],
        gamma=1,
    )
    certificate = build_global_certificate(instance, [0, 0])
    payload = certificate.to_dict()

    changed_bound = copy.deepcopy(payload)
    feasible_record = next(
        record for record in changed_bound["thresholds"] if record["feasible"]
    )
    feasible_record["U_theta"] = "999"
    check = verify_global_certificate(instance, changed_bound)
    assert not check.valid
    assert any("U_theta mismatch" in error for error in check.errors)

    missing_threshold = copy.deepcopy(payload)
    missing_threshold["thresholds"].pop()
    check = verify_global_certificate(instance, missing_threshold)
    assert not check.valid
    assert any("complete ordered original family" in error for error in check.errors)


def test_checker_rejects_instance_and_selection_tampering() -> None:
    instance = PricingInstance(
        items=[
            [Option(1.0, 3.0, 0.0), Option(2.0, -1.0, 1.0)],
            [Option(1.0, 3.0, 0.0), Option(2.0, -1.0, 1.0)],
        ],
        gamma=1,
    )
    certificate = build_global_certificate(instance, [0, 0])

    changed_instance = PricingInstance(
        items=[
            [Option(1.0, 3.0, 0.0), Option(2.0, -1.0, 1.0)],
            [Option(1.0, 3.0, 0.0), Option(2.5, -1.0, 1.0)],
        ],
        gamma=1,
    )
    check = verify_global_certificate(changed_instance, certificate)
    assert not check.valid
    assert "instance hash mismatch" in check.errors

    payload = certificate.to_dict()
    payload["selections"] = [1, 1]
    check = verify_global_certificate(instance, payload)
    assert not check.valid
    assert any("sorted-Gamma" in error for error in check.errors)


def test_negative_capacity_threshold_is_recorded_without_a_bound() -> None:
    # Gamma=0 makes the original selection robust feasible, but theta=0 has
    # negative fixed-threshold capacity; theta=1 restores positive capacity.
    instance = PricingInstance(
        items=[[Option(value=1.0, margin=0.5, uncertainty=1.0)]],
        gamma=0,
    )
    certificate = build_global_certificate(instance, [0])

    assert len(certificate.thresholds) == 2
    theta_zero = certificate.thresholds[0]
    assert theta_zero.theta == 0
    assert not theta_zero.feasible
    assert theta_zero.multiplier is None
    assert theta_zero.upper_bound is None
    assert certificate.thresholds[1].feasible
    assert verify_global_certificate(instance, certificate).valid


@pytest.mark.parametrize("gamma", [0, 3])
def test_gamma_zero_and_gamma_n(gamma: int) -> None:
    instance = PricingInstance(
        items=[
            [Option(value=float(item + 1), margin=2.0, uncertainty=1.0)]
            for item in range(3)
        ],
        gamma=gamma,
    )
    certificate = build_global_certificate(instance, [0, 0, 0])
    check = verify_global_certificate(instance, certificate)

    assert check.valid, check.errors
    assert certificate.robust_residual == Fraction(6 - gamma)
    assert len(certificate.thresholds) == 2


def test_exact_multiplier_mapping_is_checked_and_can_be_supplied() -> None:
    instance = PricingInstance(
        items=[[Option(1.0, 2.0, 0.0), Option(4.0, 1.0, 1.0)]],
        gamma=0,
    )
    candidates = original_theta_candidates(instance)
    certificate = build_global_certificate(
        instance,
        [0],
        multipliers={theta: Fraction(0) for theta in candidates},
    )
    assert verify_global_certificate(instance, certificate).valid

    with pytest.raises(ValueError, match="missing multiplier"):
        build_global_certificate(instance, [0], multipliers={candidates[-1]: 0.0})


def test_nonoptimal_multiplier_is_rejected_by_builder_and_checker() -> None:
    # At theta=0 the LP is max 10*x subject to 2*x <= 1, so lambda=5 is
    # optimal while lambda=0 gives a valid but strictly looser dual bound.
    instance = PricingInstance(
        items=[
            [
                Option(value=0.0, margin=1.0, uncertainty=0.0),
                Option(value=10.0, margin=-1.0, uncertainty=0.0),
            ]
        ],
        gamma=0,
    )
    certificate = build_global_certificate(instance, [0])
    assert certificate.thresholds[0].multiplier == Fraction(5)
    assert certificate.u == Fraction(5)

    with pytest.raises(ValueError, match="not dual optimal"):
        build_global_certificate(
            instance,
            [0],
            multipliers={Fraction(0): Fraction(0)},
        )

    # Make every serialized derived value consistent with the looser lambda=0
    # evaluation.  The verifier must still reject it because this is U-hat, not
    # the sharp LP upper bound U promised by the schema.
    payload = certificate.to_dict()
    payload["thresholds"][0]["lambda"] = "0"
    payload["thresholds"][0]["U_theta"] = "10"
    payload["U"] = "10"
    payload["absolute_gap"] = "10"
    payload["relative_width"] = "1"
    check = verify_global_certificate(instance, payload)
    assert not check.valid
    assert any("nonoptimal multiplier" in error for error in check.errors)


def test_zero_capacity_flat_dual_ray_is_accepted_exactly() -> None:
    instance = PricingInstance(
        items=[[Option(value=2.0, margin=0.0, uncertainty=0.0)]],
        gamma=0,
    )
    certificate = build_global_certificate(
        instance,
        [0],
        multipliers={Fraction(0): Fraction(17)},
    )
    assert certificate.u == Fraction(2)
    assert verify_global_certificate(instance, certificate).valid


@pytest.mark.parametrize("objective", [-1.0, 0.0])
def test_relative_width_is_undefined_for_nonpositive_u(objective: float) -> None:
    instance = PricingInstance(
        items=[[Option(value=objective, margin=1.0, uncertainty=0.0)]],
        gamma=0,
    )
    certificate = build_global_certificate(instance, [0])

    assert certificate.u == Fraction.from_float(objective)
    assert certificate.relative_width is None
    check = verify_global_certificate(instance, certificate)
    assert check.valid, check.errors
    assert check.exact_relative_width is None


def test_json_roundtrip_preserves_all_exact_rationals() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(0.1, 0.3, 0.2),
                Option(0.4, 0.7, 0.6),
            ],
            [Option(0.9, 1.1, 0.8)],
        ],
        gamma=1,
    )
    certificate = build_global_certificate(instance, [0, 0])
    raw = certificate.to_json(indent=None)
    restored = GlobalDualCertificate.from_json(raw)

    assert restored == certificate
    assert json.loads(restored.to_json(indent=None)) == json.loads(raw)
    assert restored.instance_hash == instance_sha256(instance)
    assert verify_global_certificate(instance, raw).valid


def _fractional_provenance_instance() -> PricingInstance:
    # At theta=0 the exact LP is max 10*x subject to 2*x <= 1.  Its basic
    # optimum mixes the two adjacent hull endpoints with upper weight 1/2.
    return PricingInstance(
        items=[
            [
                Option(value=0.0, margin=1.0, uncertainty=0.0),
                Option(value=10.0, margin=-1.0, uncertainty=0.0),
            ]
        ],
        gamma=0,
    )


def test_optional_provenance_carries_exact_lp_basis_and_literal_rounding() -> None:
    instance = _fractional_provenance_instance()
    legacy = build_global_certificate(instance, [0])
    certificate = build_global_certificate(
        instance,
        [0],
        include_rounding_provenance=True,
    )

    assert legacy.schema_version == CERTIFICATE_SCHEMA_VERSION
    assert "rounding_provenance" not in legacy.to_dict()
    assert verify_global_certificate(instance, legacy).valid

    assert certificate.schema_version == ROUNDING_PROVENANCE_SCHEMA_VERSION
    provenance = certificate.rounding_provenance
    assert provenance is not None
    assert provenance.theta == 0
    assert provenance.fractional_group == 0
    assert provenance.fractional_lower_option == 0
    assert provenance.fractional_upper_option == 1
    assert provenance.mixing_fraction == Fraction(1, 2)
    assert provenance.groups[0].upper_weight == Fraction(1, 2)
    assert provenance.primal_cost == 1
    assert provenance.primal_value == certificate.u == 5
    assert provenance.rounded_selections == (0,)
    assert provenance.rounded_value == certificate.h == 0
    assert provenance.rounding_loss == 5
    assert provenance.adjacent_value_jump == 10

    check = verify_global_certificate(instance, certificate.to_json(indent=None))
    assert check.valid, check.errors
    assert check.rounding_provenance_verified
    assert check.exact_rounding_loss == 5


@pytest.mark.parametrize(
    ("field_path", "replacement", "expected_error"),
    [
        (
            ("mixing_fraction",),
            "1/3",
            "mixing fraction metadata",
        ),
        (
            ("groups", 0, "upper_weight"),
            "1/3",
            "primal-dual equality fails",
        ),
        (
            ("rounded_selections",),
            [1],
            "literal rounded selection",
        ),
        (
            ("primal_value",),
            "6",
            "stored primal value",
        ),
        (
            ("adjacent_value_jump",),
            "9",
            "stored adjacent value jump",
        ),
    ],
)
def test_checker_rejects_rounding_provenance_corruption(
    field_path: tuple[object, ...],
    replacement: object,
    expected_error: str,
) -> None:
    instance = _fractional_provenance_instance()
    certificate = build_global_certificate(
        instance,
        [0],
        include_rounding_provenance=True,
    )
    payload = certificate.to_dict()
    target: object = payload["rounding_provenance"]
    for component in field_path[:-1]:
        target = target[component]  # type: ignore[index]
    target[field_path[-1]] = replacement  # type: ignore[index]

    check = verify_global_certificate(instance, payload)
    assert not check.valid
    assert not check.rounding_provenance_verified
    assert any(expected_error in error for error in check.errors), check.errors


def test_checker_rejects_provenance_at_a_nonmaximizing_threshold() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(value=0.0, margin=2.0, uncertainty=0.0),
                Option(value=10.0, margin=0.0, uncertainty=1.0),
            ]
        ],
        gamma=1,
    )
    certificate = build_global_certificate(
        instance,
        [0],
        include_rounding_provenance=True,
    )
    assert certificate.rounding_provenance is not None
    assert certificate.rounding_provenance.theta == 0
    payload = certificate.to_dict()
    payload["rounding_provenance"]["theta"] = "1"

    check = verify_global_certificate(instance, payload)
    assert not check.valid
    assert any(
        "does not attain the global LP maximum" in error for error in check.errors
    ), check.errors


def test_builder_requires_h_to_dominate_requested_rounding_provenance() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(value=0.0, margin=1.0, uncertainty=0.0),
                Option(value=2.0, margin=1.0, uncertainty=0.0),
                Option(value=10.0, margin=-1.0, uncertainty=0.0),
            ]
        ],
        gamma=0,
    )

    with pytest.raises(ValueError, match="does not dominate"):
        build_global_certificate(
            instance,
            [0],
            include_rounding_provenance=True,
        )

    certificate = build_global_certificate(
        instance,
        [1],
        include_rounding_provenance=True,
    )
    check = verify_global_certificate(instance, certificate)
    assert check.valid, check.errors
    assert check.rounding_provenance_verified


def test_integral_provenance_has_null_fractional_metadata_and_zero_loss() -> None:
    instance = PricingInstance(
        items=[
            [
                Option(value=1.0, margin=1.0, uncertainty=0.0),
                Option(value=3.0, margin=2.0, uncertainty=0.0),
            ],
            [Option(value=4.0, margin=1.0, uncertainty=0.0)],
        ],
        gamma=0,
    )
    certificate = build_global_certificate(
        instance,
        [1, 0],
        include_rounding_provenance=True,
    )
    provenance = certificate.rounding_provenance
    assert provenance is not None
    assert provenance.fractional_group is None
    assert provenance.fractional_lower_option is None
    assert provenance.fractional_upper_option is None
    assert provenance.mixing_fraction is None
    assert provenance.rounding_loss == 0
    assert provenance.adjacent_value_jump == 0
    assert verify_global_certificate(instance, certificate).valid


def test_schema_versions_reject_mixed_base_and_provenance_shapes() -> None:
    instance = _fractional_provenance_instance()
    legacy = build_global_certificate(instance, [0]).to_dict()
    extended = build_global_certificate(
        instance,
        [0],
        include_rounding_provenance=True,
    ).to_dict()

    legacy["rounding_provenance"] = copy.deepcopy(
        extended["rounding_provenance"]
    )
    check = verify_global_certificate(instance, legacy)
    assert not check.valid
    assert any("malformed certificate" in error for error in check.errors)

    extended.pop("rounding_provenance")
    check = verify_global_certificate(instance, extended)
    assert not check.valid
    assert any("malformed certificate" in error for error in check.errors)
