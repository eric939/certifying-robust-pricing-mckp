from __future__ import annotations

import copy

import numpy as np
import pytest

from robust_mckp import instance_sha256, solve_reference as solve
from scripts import run_v3_independent_rq_campaign as campaign
from scripts import verify_v3_independent_rq_release as release_verifier


def test_v3_generator_separates_reference_cost_and_surplus_signs() -> None:
    raw = campaign.generate_pricing_data(
        seed=5101, n=12, m=8, regime="favorable"
    )
    assert np.all(raw.reference_prices != raw.unit_costs)
    assert np.all(raw.unit_costs < raw.reference_prices)

    instance = campaign.build_instance(raw, gamma=3, radius=0.12)
    negative, positive = campaign._mixed_sign_counts(instance)
    assert negative > 0
    assert positive > 0


def test_v3_floating_sweep_sandwich_and_rounding_jump() -> None:
    raw = campaign.generate_pricing_data(
        seed=5102, n=7, m=5, regime="favorable"
    )
    instance = campaign.build_instance(raw, gamma=2, radius=0.12)
    solution = solve(instance, upgrade_completion=True)
    sweep = campaign.floating_full_family_interval(instance)

    assert solution.is_feasible
    assert sweep["U_float"] + 1e-7 >= solution.objective
    assert sweep["literal_H_float"] <= sweep["U_float"] + 1e-7
    assert (
        sweep["rounding_loss_at_theta_U"]
        <= sweep["fractional_jump_at_theta_U"] + 1e-6
    )


def test_v3_instance_snapshot_round_trip() -> None:
    raw = campaign.generate_pricing_data(
        seed=6101, n=8, m=3, regime="adversarial"
    )
    instance = campaign.build_instance(raw, gamma=2, radius=0.22)
    payload = campaign.instance_payload(
        instance, raw, gamma_mode="sqrt", radius=0.22
    )
    restored = campaign.load_instance_payload(payload)
    assert instance_sha256(restored) == instance_sha256(instance)


def test_v3_release_verifier_reconstructs_primitive_pricing_coefficients() -> None:
    raw = campaign.generate_pricing_data(
        seed=5101, n=6, m=5, regime="favorable"
    )
    instance = campaign.build_instance(raw, gamma=2, radius=0.12)
    payload = campaign.instance_payload(
        instance, raw, gamma_mode="sqrt", radius=0.12
    )
    release_verifier.verify_pricing_payload(payload)

    corrupted = copy.deepcopy(payload)
    corrupted_cost = 0.99 * float.fromhex(corrupted["unit_costs"][0])
    corrupted["unit_costs"][0] = corrupted_cost.hex()
    with pytest.raises(SystemExit, match="margin from"):
        release_verifier.verify_pricing_payload(corrupted)


def test_v3_release_verifier_rejects_recomputed_summary_drift() -> None:
    expected = {
        "coverage": {
            "rate": 0.9,
            "wilson": [0.81, 0.95],
        }
    }
    actual = copy.deepcopy(expected)
    release_verifier.require_equivalent(actual, expected, "summary")
    actual["coverage"]["wilson"][0] = 0.80
    with pytest.raises(SystemExit, match=r"summary.coverage.wilson\[0\] mismatch"):
        release_verifier.require_equivalent(actual, expected, "summary")


def test_v3_calibration_stream_is_deterministic_and_coverage_implies_feasibility() -> None:
    seed = 710017
    first = campaign._draw_residual_path(
        cell="exchangeable_correlated",
        rng=np.random.default_rng(np.random.SeedSequence([seed, 29082026])),
        n=12,
        length=101,
    )
    second = campaign._draw_residual_path(
        cell="exchangeable_correlated",
        rng=np.random.default_rng(np.random.SeedSequence([seed, 29082026])),
        n=12,
        length=101,
    )
    assert np.array_equal(first, second)

    gamma = campaign._tune_gamma(first[:40])
    radius = campaign._conformal_radius(
        [campaign._uncertainty_score(row, gamma) for row in first[40:100]],
        campaign.ALPHA,
    )
    test = first[-1].copy()
    score = campaign._uncertainty_score(test, gamma)
    if score > radius:
        test *= 0.99 * radius / score
    assert campaign._uncertainty_score(test, gamma) <= radius

    raw = campaign.generate_pricing_data(
        seed=seed, n=12, m=5, regime="favorable"
    )
    instance = campaign.build_instance(raw, gamma=gamma, radius=radius)
    solution = solve(instance, upgrade_completion=True)
    realized = campaign._realized_policy_metrics(raw, solution.selections, test)
    assert realized["realized_margin"] >= -1e-7 * solution.objective
