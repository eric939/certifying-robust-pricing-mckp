from __future__ import annotations

import itertools

import pytest

from robust_mckp import GlobalThetaBNBConfig, Option, PricingInstance
from robust_mckp.certificate import compute_certificate
from robust_mckp.exact_bnb import solve_global_theta_bnb
from robust_mckp.local_budget import (
    SegmentLocalExactConfig,
    build_local_theta_candidates,
    robust_certificate_segment_local,
    solve_segment_local_exact,
)


def _objective(instance: PricingInstance, selection) -> float:
    return float(sum(instance.items[i][int(j)].value for i, j in enumerate(selection)))


def _bruteforce_local(instance: PricingInstance, segments, gammas):
    best = None
    best_obj = float("-inf")
    for combo in itertools.product(*[range(len(group)) for group in instance.items]):
        cert = robust_certificate_segment_local(instance, combo, segments, gammas)
        if cert < -1e-9:
            continue
        obj = _objective(instance, combo)
        if obj > best_obj + 1e-9:
            best = list(map(int, combo))
            best_obj = obj
    return best, best_obj


def _tiny_instance() -> PricingInstance:
    return PricingInstance(
        items=[
            [Option(0.0, 4.0, 0.0), Option(8.0, 1.0, 2.0)],
            [Option(1.0, 3.0, 0.0), Option(7.0, 0.0, 3.0)],
            [Option(0.0, 2.0, 0.0), Option(6.0, -1.0, 1.0)],
        ],
        gamma=1,
    )


def test_one_segment_local_budget_equals_global_gamma() -> None:
    instance = _tiny_instance()
    segments = ["all"] * instance.n_items
    gammas = {"all": instance.gamma}
    local = solve_segment_local_exact(instance, segments, gammas)
    global_res = solve_global_theta_bnb(instance, GlobalThetaBNBConfig(use_hullround_incumbent=False))
    assert local.status == global_res.status == "optimal"
    assert local.objective_value == pytest.approx(global_res.objective_value, abs=1e-8)
    assert robust_certificate_segment_local(instance, local.selected_options, segments, gammas) == pytest.approx(
        compute_certificate(instance, local.selected_options), abs=1e-8
    )


def test_two_segment_local_budget_matches_bruteforce() -> None:
    instance = _tiny_instance()
    segments = ["a", "a", "b"]
    gammas = {"a": 1, "b": 0}
    exact = solve_segment_local_exact(instance, segments, gammas)
    brute_sel, brute_obj = _bruteforce_local(instance, segments, gammas)
    assert exact.status == "optimal"
    assert exact.objective_value == pytest.approx(brute_obj, abs=1e-8)
    assert exact.selected_options is not None
    assert _objective(instance, exact.selected_options) == pytest.approx(_objective(instance, brute_sel), abs=1e-8)
    assert exact.robust_certificate >= -1e-8


def test_three_segment_local_budget_matches_bruteforce() -> None:
    instance = PricingInstance(
        items=[
            [Option(0.0, 2.0, 0.0), Option(3.0, 0.0, 2.0)],
            [Option(0.0, 2.0, 0.0), Option(4.0, 0.0, 1.0)],
            [Option(0.0, 2.0, 0.0), Option(5.0, 0.0, 1.0)],
        ],
        gamma=0,
    )
    segments = ["a", "b", "c"]
    gammas = {"a": 0, "b": 1, "c": 1}
    exact = solve_segment_local_exact(instance, segments, gammas)
    _sel, brute_obj = _bruteforce_local(instance, segments, gammas)
    assert exact.status == "optimal"
    assert exact.objective_value == pytest.approx(brute_obj, abs=1e-8)


def test_product_size_guard_returns_clear_status() -> None:
    instance = _tiny_instance()
    segments = ["a", "b", "c"]
    gammas = {"a": 0, "b": 0, "c": 0}
    result = solve_segment_local_exact(
        instance,
        segments,
        gammas,
        SegmentLocalExactConfig(max_theta_vectors=1),
    )
    assert result.status == "too_many_theta_vectors"
    assert "exceeds max_theta_vectors" in result.message


def test_invalid_segment_inputs_raise_clean_errors() -> None:
    instance = _tiny_instance()
    with pytest.raises(ValueError, match="segments length"):
        solve_segment_local_exact(instance, ["a"], {"a": 0})
    with pytest.raises(ValueError, match="missing segment"):
        solve_segment_local_exact(instance, ["a", "a", "b"], {"a": 1})
    with pytest.raises(ValueError, match="must be in"):
        solve_segment_local_exact(instance, ["a", "a", "b"], {"a": 3, "b": 0})


def test_build_local_theta_candidates_preserves_segment_breakpoints() -> None:
    instance = _tiny_instance()
    candidates = build_local_theta_candidates(instance, ["a", "a", "b"])
    assert candidates["a"] == pytest.approx([0.0, 2.0, 3.0])
    assert candidates["b"] == pytest.approx([0.0, 1.0])


def test_local_theta_candidates_preserve_binary64_distinct_nearby_values() -> None:
    delta = 9e-11
    instance = PricingInstance(
        items=[
            [Option(1.0, 1.0, 1.0)],
            [Option(1.0, 1.0, 1.0 + delta)],
        ],
        gamma=1,
    )
    candidates = build_local_theta_candidates(
        instance,
        ["a", "a"],
        tol=1e-9,
    )
    assert candidates["a"] == [0.0, 1.0, 1.0 + delta]
