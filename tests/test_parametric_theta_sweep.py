from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from robust_mckp import GlobalThetaBNBConfig, Option, PricingInstance
from robust_mckp.exact_bnb import (
    _build_fixed_theta_cache,
    brute_force_global_robust,
    build_fixed_theta_data,
    compute_fixed_theta_lp_upper_bound,
    build_full_theta_candidates,
    solve_global_theta_bnb,
)
from robust_mckp.milp_baselines import solve_theta_decomposition_milp_baseline
from robust_mckp.parametric_sweep import (
    ParametricThetaSweepConfig,
    build_parametric_theta_sweep,
    solve_global_theta_bnb_sweep,
)


def _random_instance(seed: int, n: int = 5, m: int = 4, gamma: int = 2) -> PricingInstance:
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        group = []
        for _j in range(m):
            group.append(
                Option(
                    value=float(rng.integers(0, 40)),
                    margin=float(rng.integers(-5, 14)),
                    uncertainty=float(rng.integers(-6, 7)),
                )
            )
        group[0] = Option(value=float(rng.integers(0, 20)), margin=float(rng.integers(3, 12)), uncertainty=0.0)
        items.append(group)
    return PricingInstance(items=items, gamma=gamma)


def _bruteforce_obj(instance: PricingInstance) -> float:
    brute = brute_force_global_robust(instance)
    return brute.objective_value if brute.status == "optimal" else float("-inf")


def test_incremental_theta_state_equals_independent_recomputation() -> None:
    for seed in range(5):
        instance = _random_instance(seed)
        sweep = build_parametric_theta_sweep(
            instance,
            config=ParametricThetaSweepConfig(validate_against_recompute=True, force_rebuild_hulls=True),
        )
        for state in sweep.states:
            direct = build_fixed_theta_data(instance, state.theta)
            assert state.data.capacity == pytest.approx(direct.capacity, abs=1e-8)
            assert state.data.baseline_indices == direct.baseline_indices
            direct_cache = _build_fixed_theta_cache(instance, state.theta, 1e-9, data=direct)
            assert state.cache.option_sets == direct_cache.option_sets
            for a, b in zip(state.data.s_theta, direct.s_theta):
                np.testing.assert_allclose(a, b, atol=1e-8, rtol=0.0)
            for a, b in zip(state.data.costs, direct.costs):
                np.testing.assert_allclose(a, b, atol=1e-8, rtol=0.0)


def test_sweep_visits_full_original_breakpoint_set() -> None:
    instance = _random_instance(111, n=4, m=5, gamma=2)
    expected = build_full_theta_candidates(instance)
    sweep = build_parametric_theta_sweep(instance, config=ParametricThetaSweepConfig())
    observed = [state.theta for state in sweep.states]
    assert observed == pytest.approx(expected, abs=1e-12)
    assert sweep.diagnostics["theta_count_candidate_total"] == len(expected)
    assert sweep.diagnostics["theta_count_states_evaluated"] == len(expected)


def test_sweep_preserves_and_reconstructs_nearby_distinct_breakpoints() -> None:
    delta = 9e-11
    instance = PricingInstance(
        items=[
            [Option(1.0, 0.5, 1.0)],
            [Option(1.0, 0.5 + delta, 1.0 + delta)],
        ],
        gamma=1,
    )
    sweep = build_parametric_theta_sweep(
        instance,
        tol=1e-9,
        config=ParametricThetaSweepConfig(
            tol=1e-9,
            validate_against_recompute=True,
            reuse_hulls=True,
        ),
    )
    assert [state.theta for state in sweep.states] == [0.0, 1.0, 1.0 + delta]
    for state in sweep.states:
        direct = build_fixed_theta_data(instance, state.theta)
        for actual, expected in zip(state.data.s_theta, direct.s_theta):
            np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0.0)


def test_hull_reuse_safety_with_validation_enabled() -> None:
    instance = PricingInstance(
        items=[
            [Option(1.0, 4.0, 0.0), Option(2.0, 2.0, 1.0)],
            [Option(3.0, 3.0, 0.0), Option(4.0, 1.0, 2.0)],
        ],
        gamma=1,
    )
    sweep = build_parametric_theta_sweep(
        instance,
        config=ParametricThetaSweepConfig(validate_against_recompute=True, reuse_hulls=True),
    )
    assert sweep.diagnostics["hull_rebuilds_total"] > 0
    assert sweep.diagnostics["hull_reuses_total"] > 0
    assert sweep.diagnostics["max_abs_root_lp_recompute_error"] <= 1e-8


def test_hull_rebuild_when_cost_geometry_changes() -> None:
    instance = PricingInstance(
        items=[
            [Option(0.0, 4.0, 0.0), Option(10.0, 1.0, 3.0)],
            [Option(0.0, 3.0, 0.0), Option(8.0, 0.0, 2.0)],
        ],
        gamma=1,
    )
    sweep = build_parametric_theta_sweep(
        instance,
        config=ParametricThetaSweepConfig(validate_against_recompute=True, reuse_hulls=True),
    )
    assert any(r.hull_rebuilds == instance.n_items for r in sweep.records[1:])


def test_lp_root_bound_parity_for_every_theta() -> None:
    instance = _random_instance(123, n=4, m=4, gamma=2)
    sweep = build_parametric_theta_sweep(
        instance,
        config=ParametricThetaSweepConfig(validate_against_recompute=True, reuse_hulls=True),
    )
    for state in sweep.states:
        direct = compute_fixed_theta_lp_upper_bound(instance, state.theta)
        assert state.lp_bound.root_lp_status == direct.root_lp_status
        if math.isfinite(state.lp_bound.lp_upper_bound) and math.isfinite(direct.lp_upper_bound):
            assert state.lp_bound.lp_upper_bound == pytest.approx(direct.lp_upper_bound, abs=1e-8)


def test_global_sweep_matches_global_enumeration_and_bruteforce() -> None:
    for seed in range(4):
        instance = _random_instance(seed + 50, n=5, m=3, gamma=2)
        cfg = GlobalThetaBNBConfig(use_hullround_incumbent=True, theta_order="increasing")
        enum = solve_global_theta_bnb(instance, cfg)
        sweep = solve_global_theta_bnb_sweep(
            instance,
            cfg,
            ParametricThetaSweepConfig(validate_against_recompute=True, reuse_hulls=True),
        )
        brute = brute_force_global_robust(instance)
        assert sweep.status == enum.status == brute.status
        if brute.status == "optimal":
            assert sweep.objective_value == pytest.approx(enum.objective_value, abs=1e-8)
            assert sweep.objective_value == pytest.approx(brute.objective_value, abs=1e-8)
            assert sweep.robust_certificate >= -1e-8
            assert sweep.absolute_gap == pytest.approx(0.0, abs=1e-8)


def test_sweep_limited_run_does_not_claim_false_optimality() -> None:
    instance = _random_instance(222, n=6, m=4, gamma=2)
    result = solve_global_theta_bnb_sweep(
        instance,
        GlobalThetaBNBConfig(use_hullround_incumbent=False, node_limit=0),
        ParametricThetaSweepConfig(validate_against_recompute=True, max_recompute_checks=1),
    )
    assert result.status == "node_limit"
    assert result.status != "optimal"
    assert result.upper_bound >= result.lower_bound
    assert result.absolute_gap >= 0.0
    assert result.diagnostics["validation_mode"] == "sampled"


def test_exact_sweep_matches_bruteforce_tiny_grid() -> None:
    instance = PricingInstance(
        items=[
            [Option(0.0, 3.0, 0.0), Option(5.0, 1.0, 2.0)],
            [Option(0.0, 3.0, 0.0), Option(6.0, 1.0, 1.0)],
            [Option(0.0, 2.0, 0.0), Option(4.0, -1.0, 2.0)],
        ],
        gamma=1,
    )
    sweep = solve_global_theta_bnb_sweep(
        instance,
        GlobalThetaBNBConfig(use_hullround_incumbent=False),
        ParametricThetaSweepConfig(validate_against_recompute=True),
    )
    assert sweep.status == "optimal"
    assert sweep.objective_value == pytest.approx(_bruteforce_obj(instance))


def test_sweep_preserves_below_hull_integer_option() -> None:
    # At theta=0 this encodes a fixed-theta MCKP where option 1 is below the
    # upper hull between options 0 and 2, but is the integer optimum at capacity 5.
    instance = PricingInstance(
        items=[
            [Option(0.0, 2.5, 0.0), Option(8.0, -2.5, 0.0), Option(20.0, -7.5, 0.0)],
            [Option(0.0, 2.5, 0.0)],
        ],
        gamma=0,
    )
    sweep = solve_global_theta_bnb_sweep(
        instance,
        GlobalThetaBNBConfig(use_hullround_incumbent=False),
        ParametricThetaSweepConfig(validate_against_recompute=True),
    )
    assert sweep.status == "optimal"
    assert sweep.objective_value == pytest.approx(8.0)
    assert sweep.selected_options == [1, 0]


def test_optional_baselines_missing_backends_are_clean() -> None:
    for backend in ["scip", "gurobi", "cplex", "does_not_exist"]:
        result = solve_theta_decomposition_milp_baseline(_random_instance(9, n=2, m=2, gamma=1), backend=backend)
        if result.available:
            assert result.status in {"optimal", "limited", "infeasible"}
        else:
            assert result.status == "not_available"


def test_scipy_highs_baseline_if_available() -> None:
    scipy = pytest.importorskip("scipy.optimize")
    if not hasattr(scipy, "milp"):
        pytest.skip("scipy.optimize.milp is unavailable")
    instance = PricingInstance(
        items=[
            [Option(0.0, 3.0, 0.0), Option(5.0, 1.0, 1.0)],
            [Option(0.0, 3.0, 0.0), Option(6.0, 1.0, 1.0)],
        ],
        gamma=1,
    )
    result = solve_theta_decomposition_milp_baseline(instance, backend="scipy_highs", time_limit_per_theta=5.0)
    assert result.available
    assert result.status == "optimal"
    assert result.objective_value == pytest.approx(_bruteforce_obj(instance), abs=1e-8)
