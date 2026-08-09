"""Exact parametric theta sweep for global Gamma-robust MCKP solving."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .certificate import compute_certificate
from .exact_bnb import (
    CachedHullSegment,
    FixedThetaBNBConfig,
    FixedThetaCache,
    FixedThetaData,
    FixedThetaLPBoundResult,
    GlobalThetaBNBConfig,
    GlobalThetaBNBResult,
    GlobalThetaRecord,
    _build_fixed_theta_cache,
    _global_result,
    build_fixed_theta_data,
    build_full_theta_candidates,
    compute_fixed_theta_lp_upper_bound,
    nondominated_option_indices,
    solve_fixed_theta_bnb,
    validate_fixed_theta_selection,
)
from .hull import Hull, build_upper_hull
from .model import PricingInstance
from .solver import solve as solve_hullround
from .utils import EPS


@dataclass(frozen=True)
class ParametricThetaSweepConfig:
    """Configuration for exact increasing-theta sweep construction."""

    tol: float = 1e-9
    validate_against_recompute: bool = False
    reuse_hulls: bool = True
    force_rebuild_hulls: bool = False
    max_recompute_checks: Optional[int] = None
    collect_diagnostics: bool = True


@dataclass
class ParametricThetaRecord:
    """Diagnostics for one theta state in the parametric sweep."""

    theta: float
    capacity: float
    s_star_sum: float
    hull_rebuilds: int
    hull_reuses: int
    item_hulls_changed: int
    item_hulls_unchanged: int
    root_lp_upper_bound: float
    root_lp_status: str
    root_lp_time_seconds: float
    update_time_seconds: float = 0.0
    baseline_time_seconds: float = 0.0
    hull_time_seconds: float = 0.0
    max_abs_s_theta_recompute_error: float = 0.0
    max_abs_cost_recompute_error: float = 0.0
    max_abs_capacity_recompute_error: float = 0.0
    max_abs_root_lp_recompute_error: float = 0.0
    validation_checked: bool = False
    diagnostics: Dict[str, object] = field(default_factory=dict)


@dataclass
class ParametricThetaState:
    """Exact fixed-theta state maintained by the sweep."""

    theta: float
    data: FixedThetaData
    cache: FixedThetaCache
    lp_bound: FixedThetaLPBoundResult
    record: ParametricThetaRecord


@dataclass
class ParametricThetaSweepResult:
    """Materialized exact sweep result."""

    states: List[ParametricThetaState]
    records: List[ParametricThetaRecord]
    diagnostics: Dict[str, object]


def _extract_arrays(instance: PricingInstance) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    values: List[np.ndarray] = []
    margins: List[np.ndarray] = []
    abs_uncertainties: List[np.ndarray] = []
    for group in instance.items:
        values.append(np.array([opt.value for opt in group], dtype=float))
        margins.append(np.array([opt.margin for opt in group], dtype=float))
        abs_uncertainties.append(np.array([abs(opt.uncertainty) for opt in group], dtype=float))
    return values, margins, abs_uncertainties


def _copy_s_theta(values: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [np.array(v, dtype=float, copy=True) for v in values]


def _build_data_from_s_theta(
    values: Sequence[np.ndarray],
    s_theta: Sequence[np.ndarray],
    gamma: int,
    theta: float,
) -> FixedThetaData:
    value_arrays: List[np.ndarray] = []
    s_theta_arrays: List[np.ndarray] = []
    costs: List[np.ndarray] = []
    baseline_indices: List[int] = []
    s_star_sum = 0.0
    for v, st in zip(values, s_theta):
        v_arr = np.array(v, dtype=float, copy=True)
        st_arr = np.array(st, dtype=float, copy=True)
        baseline_idx = int(np.argmax(st_arr))
        s_star = float(st_arr[baseline_idx])
        value_arrays.append(v_arr)
        s_theta_arrays.append(st_arr)
        costs.append(np.maximum(s_star - st_arr, 0.0))
        baseline_indices.append(baseline_idx)
        s_star_sum += s_star
    return FixedThetaData(
        values=value_arrays,
        s_theta=s_theta_arrays,
        costs=costs,
        capacity=float(s_star_sum - float(gamma) * float(theta)),
        s_star_sum=float(s_star_sum),
        baseline_indices=baseline_indices,
    )


def _same_point_set(
    prev_costs: Optional[np.ndarray],
    costs: np.ndarray,
    prev_values: Optional[np.ndarray],
    values: np.ndarray,
    tol: float,
) -> bool:
    if prev_costs is None or prev_values is None:
        return False
    if prev_costs.shape != costs.shape or prev_values.shape != values.shape:
        return False
    del tol
    return bool(np.array_equal(prev_costs, costs) and np.array_equal(prev_values, values))


def _hulls_equal(a: Hull, b: Hull, tol: float) -> bool:
    return (
        a.costs.shape == b.costs.shape
        and a.values.shape == b.values.shape
        and a.option_indices.shape == b.option_indices.shape
        and np.allclose(a.costs, b.costs, atol=tol, rtol=0.0)
        and np.allclose(a.values, b.values, atol=tol, rtol=0.0)
        and np.array_equal(a.option_indices, b.option_indices)
    )


def _build_cache_from_sweep_data(
    data: FixedThetaData,
    theta: float,
    tol: float,
    *,
    previous_cache: Optional[FixedThetaCache],
    previous_point_costs: Sequence[Optional[np.ndarray]],
    previous_point_values: Sequence[Optional[np.ndarray]],
    reuse_hulls: bool,
    force_rebuild_hulls: bool,
    validate_reused_hulls: bool,
) -> Tuple[FixedThetaCache, int, int, int, int, List[Optional[np.ndarray]], List[Optional[np.ndarray]], float]:
    hull_start = time.perf_counter()
    option_sets = [nondominated_option_indices(c, v, tol) for v, c in zip(data.values, data.costs)]
    min_cost = [
        float(min(float(data.costs[i][j]) for j in opts)) if opts else float("inf")
        for i, opts in enumerate(option_sets)
    ]
    per_item_hulls: List[Hull] = []
    hull_base_cost: List[float] = []
    hull_base_value: List[float] = []
    max_value: List[float] = []
    global_segments: List[CachedHullSegment] = []
    next_point_costs: List[Optional[np.ndarray]] = []
    next_point_values: List[Optional[np.ndarray]] = []
    rebuilds = 0
    reuses = 0
    item_hulls_changed = 0
    item_hulls_unchanged = 0
    for i, opts in enumerate(option_sets):
        costs = np.array(data.costs[i], dtype=float, copy=True)
        values = np.array(data.values[i], dtype=float, copy=True)
        can_reuse = (
            bool(reuse_hulls)
            and not force_rebuild_hulls
            and previous_cache is not None
            and i < len(previous_cache.per_item_hulls)
            and list(previous_cache.option_sets[i]) == list(opts)
            and _same_point_set(previous_point_costs[i], costs, previous_point_values[i], values, tol)
        )
        if can_reuse:
            hull = previous_cache.per_item_hulls[i]
            if validate_reused_hulls:
                idx_check = np.array(opts, dtype=int)
                rebuilt = (
                    build_upper_hull(costs[idx_check], values[idx_check], idx_check)
                    if idx_check.size
                    else build_upper_hull(np.array([], dtype=float), np.array([], dtype=float), idx_check)
                )
                if not _hulls_equal(hull, rebuilt, tol):
                    raise ValueError(f"reused hull validation failed for item {i} at theta={theta}")
            reuses += 1
            item_hulls_unchanged += 1
        else:
            idx = np.array(opts, dtype=int)
            hull = (
                build_upper_hull(costs[idx], values[idx], idx)
                if idx.size
                else build_upper_hull(np.array([], dtype=float), np.array([], dtype=float), idx)
            )
            rebuilds += 1
            item_hulls_changed += 1
        per_item_hulls.append(hull)
        hull_base_cost.append(float(hull.costs[0]) if hull.costs.size else float("inf"))
        hull_base_value.append(float(hull.values[0]) if hull.values.size else float("-inf"))
        max_value.append(float(max(float(data.values[i][j]) for j in opts)) if opts else float("-inf"))
        for k, (slope, length) in enumerate(zip(hull.slopes, hull.delta_costs)):
            if float(length) > 0.0:
                global_segments.append(CachedHullSegment(item=i, index=k, slope=float(slope), length=float(length)))
        next_point_costs.append(costs)
        next_point_values.append(values)
    global_segments.sort(key=lambda seg: (-seg.slope, int(seg.item), int(seg.index)))
    cache = FixedThetaCache(
        theta=float(theta),
        data=data,
        option_sets=option_sets,
        min_cost=min_cost,
        per_item_hulls=per_item_hulls,
        hull_base_cost=hull_base_cost,
        hull_base_value=hull_base_value,
        max_value=max_value,
        global_segments=tuple(global_segments),
    )
    return (
        cache,
        rebuilds,
        reuses,
        item_hulls_changed,
        item_hulls_unchanged,
        next_point_costs,
        next_point_values,
        float(time.perf_counter() - hull_start),
    )


def iter_parametric_theta_states(
    instance: PricingInstance,
    *,
    tol: Optional[float] = None,
    config: Optional[ParametricThetaSweepConfig] = None,
) -> Iterator[ParametricThetaState]:
    """Yield exact fixed-theta states over the full original breakpoint set.

    The iterator uses the incremental update
    ``s^{r+1}_{ij} = s^r_{ij} + (theta_{r+1}-theta_r) 1{|t_ij| > theta_r}``
    and then recomputes baselines, costs, capacity, option sets, and LP-bound
    hull structures from the updated arrays.
    """

    cfg = config or ParametricThetaSweepConfig()
    effective_tol = float(tol if tol is not None else cfg.tol)
    values, margins, abs_uncertainties = _extract_arrays(instance)
    candidate_start = time.perf_counter()
    candidates = build_full_theta_candidates(instance, tol=effective_tol)
    candidate_generation_time = time.perf_counter() - candidate_start
    s_theta = [s - np.maximum(0.0, abs_t - candidates[0]) for s, abs_t in zip(margins, abs_uncertainties)]
    previous_cache: Optional[FixedThetaCache] = None
    previous_point_costs: List[Optional[np.ndarray]] = [None for _ in values]
    previous_point_values: List[Optional[np.ndarray]] = [None for _ in values]
    recompute_checks = 0
    previous_theta = float(candidates[0])

    for idx, theta in enumerate(candidates):
        theta = float(theta)
        update_time = 0.0
        if idx > 0:
            update_start = time.perf_counter()
            delta = theta - previous_theta
            for k, abs_t in enumerate(abs_uncertainties):
                s_theta[k] = s_theta[k] + float(delta) * (abs_t > previous_theta)
            update_time = time.perf_counter() - update_start

        baseline_start = time.perf_counter()
        data = _build_data_from_s_theta(values, s_theta, instance.gamma, theta)
        baseline_time = time.perf_counter() - baseline_start

        validate_this = bool(cfg.validate_against_recompute) and (
            cfg.max_recompute_checks is None or recompute_checks < int(cfg.max_recompute_checks)
        )
        max_s_error = 0.0
        max_cost_error = 0.0
        max_capacity_error = 0.0
        if validate_this:
            direct = build_fixed_theta_data(instance, theta)
            max_s_error = max(
                [0.0]
                + [float(np.max(np.abs(a - b))) for a, b in zip(data.s_theta, direct.s_theta) if a.size]
            )
            max_cost_error = max(
                [0.0]
                + [float(np.max(np.abs(a - b))) for a, b in zip(data.costs, direct.costs) if a.size]
            )
            max_capacity_error = abs(float(data.capacity) - float(direct.capacity))
            if max(max_s_error, max_cost_error, max_capacity_error) > 10.0 * effective_tol:
                raise ValueError(
                    "parametric sweep state differs from independent recomputation "
                    f"at theta={theta}: s={max_s_error}, cost={max_cost_error}, capacity={max_capacity_error}"
                )
            recompute_checks += 1

        (
            cache,
            rebuilds,
            reuses,
            item_hulls_changed,
            item_hulls_unchanged,
            previous_point_costs,
            previous_point_values,
            hull_time,
        ) = _build_cache_from_sweep_data(
            data,
            theta,
            effective_tol,
            previous_cache=previous_cache,
            previous_point_costs=previous_point_costs,
            previous_point_values=previous_point_values,
            reuse_hulls=cfg.reuse_hulls,
            force_rebuild_hulls=cfg.force_rebuild_hulls,
            validate_reused_hulls=validate_this,
        )

        lp_bound = compute_fixed_theta_lp_upper_bound(instance, theta, effective_tol, cache=cache)
        root_recompute_error = 0.0
        if validate_this:
            rebuilt_cache = _build_fixed_theta_cache(instance, theta, effective_tol)
            direct_lp = compute_fixed_theta_lp_upper_bound(instance, theta, effective_tol, cache=rebuilt_cache)
            if math.isfinite(lp_bound.lp_upper_bound) and math.isfinite(direct_lp.lp_upper_bound):
                root_recompute_error = abs(float(lp_bound.lp_upper_bound) - float(direct_lp.lp_upper_bound))
            elif lp_bound.lp_upper_bound != direct_lp.lp_upper_bound:
                root_recompute_error = float("inf")
            if root_recompute_error > 10.0 * effective_tol:
                raise ValueError(
                    "parametric sweep root LP differs from independent recomputation "
                    f"at theta={theta}: {root_recompute_error}"
                )

        record = ParametricThetaRecord(
            theta=theta,
            capacity=float(data.capacity),
            s_star_sum=float(data.s_star_sum),
            hull_rebuilds=int(rebuilds),
            hull_reuses=int(reuses),
            item_hulls_changed=int(item_hulls_changed),
            item_hulls_unchanged=int(item_hulls_unchanged),
            root_lp_upper_bound=float(lp_bound.lp_upper_bound),
            root_lp_status=lp_bound.root_lp_status,
            root_lp_time_seconds=float(lp_bound.runtime_seconds),
            update_time_seconds=float(update_time),
            baseline_time_seconds=float(baseline_time),
            hull_time_seconds=float(hull_time),
            max_abs_s_theta_recompute_error=float(max_s_error),
            max_abs_cost_recompute_error=float(max_cost_error),
            max_abs_capacity_recompute_error=float(max_capacity_error),
            max_abs_root_lp_recompute_error=float(root_recompute_error),
            validation_checked=validate_this,
            diagnostics={
                "theta_candidate_source": "full_original_breakpoints",
                "candidate_generation_time_seconds": float(candidate_generation_time if idx == 0 else 0.0),
                "item_hulls_changed": int(item_hulls_changed),
                "item_hulls_unchanged": int(item_hulls_unchanged),
                "dirty_item_fraction": float(item_hulls_changed / max(1, item_hulls_changed + item_hulls_unchanged)),
            },
        )
        state = ParametricThetaState(theta=theta, data=data, cache=cache, lp_bound=lp_bound, record=record)
        yield state
        previous_cache = cache
        previous_theta = theta


def build_parametric_theta_sweep(
    instance: PricingInstance,
    *,
    tol: Optional[float] = None,
    config: Optional[ParametricThetaSweepConfig] = None,
) -> ParametricThetaSweepResult:
    """Materialize the exact parametric theta sweep and aggregate diagnostics."""

    cfg = config or ParametricThetaSweepConfig()
    states = list(iter_parametric_theta_states(instance, tol=tol, config=cfg))
    records = [state.record for state in states]
    rebuilds = sum(r.hull_rebuilds for r in records)
    reuses = sum(r.hull_reuses for r in records)
    item_hulls_changed = sum(r.item_hulls_changed for r in records)
    item_hulls_unchanged = sum(r.item_hulls_unchanged for r in records)
    candidate_count = len(build_full_theta_candidates(instance, tol=float(tol if tol is not None else cfg.tol)))
    validation_checks = sum(1 for r in records if r.validation_checked)
    validation_mode = "off"
    if cfg.validate_against_recompute:
        validation_mode = "exhaustive" if cfg.max_recompute_checks is None else "sampled"
    diagnostics = {
        "theta_count_total": len(records),
        "theta_count_candidate_total": int(candidate_count),
        "theta_candidate_source": "full_original_breakpoints",
        "theta_count_states_evaluated": len(records),
        "theta_values": [float(r.theta) for r in records],
        "candidate_generation_time_seconds": float(
            sum(float(r.diagnostics.get("candidate_generation_time_seconds", 0.0)) for r in records)
        ),
        "sweep_update_time_seconds": float(sum(r.update_time_seconds for r in records)),
        "baseline_update_time_seconds": float(sum(r.baseline_time_seconds for r in records)),
        "hull_rebuilds_total": int(rebuilds),
        "hull_reuses_total": int(reuses),
        "item_hulls_changed_total": int(item_hulls_changed),
        "item_hulls_unchanged_total": int(item_hulls_unchanged),
        "dirty_item_fraction": float(item_hulls_changed / max(1, item_hulls_changed + item_hulls_unchanged)),
        "hull_reuse_rate": float(reuses / max(1, rebuilds + reuses)),
        "hull_rebuilds_by_theta": [int(r.hull_rebuilds) for r in records],
        "hull_reuses_by_theta": [int(r.hull_reuses) for r in records],
        "root_lp_time_seconds": float(sum(r.root_lp_time_seconds for r in records)),
        "max_abs_s_theta_recompute_error": float(max([0.0] + [r.max_abs_s_theta_recompute_error for r in records])),
        "max_abs_cost_recompute_error": float(max([0.0] + [r.max_abs_cost_recompute_error for r in records])),
        "max_abs_capacity_recompute_error": float(max([0.0] + [r.max_abs_capacity_recompute_error for r in records])),
        "max_abs_root_lp_recompute_error": float(max([0.0] + [r.max_abs_root_lp_recompute_error for r in records])),
        "validation_against_recompute": bool(cfg.validate_against_recompute),
        "validation_mode": validation_mode,
        "validation_checks_performed": int(validation_checks),
        "validation_max_recompute_checks": cfg.max_recompute_checks,
        "hull_reuse_enabled": bool(cfg.reuse_hulls and not cfg.force_rebuild_hulls),
    }
    return ParametricThetaSweepResult(states=states, records=records, diagnostics=diagnostics)


def solve_global_theta_bnb_sweep(
    instance: PricingInstance,
    config: Optional[GlobalThetaBNBConfig] = None,
    sweep_config: Optional[ParametricThetaSweepConfig] = None,
) -> GlobalThetaBNBResult:
    """Solve the global robust MCKP with exact increasing-theta sweep data."""

    cfg = config or GlobalThetaBNBConfig()
    sw_cfg = sweep_config or ParametricThetaSweepConfig(tol=cfg.tolerance)
    tol = float(cfg.tolerance)
    start = time.perf_counter()
    records: List[GlobalThetaRecord] = []
    total_nodes = 0
    total_root_lp_time = 0.0
    total_bnb_time = 0.0
    theta_upper_bounds: List[float] = []
    diagnostics: Dict[str, object] = {
        "theta_mode": "sweep",
        "theta_order": "increasing",
        "hull_reuse_enabled": bool(sw_cfg.reuse_hulls and not sw_cfg.force_rebuild_hulls),
        "validation_against_recompute": bool(sw_cfg.validate_against_recompute),
    }
    candidate_count_total = len(build_full_theta_candidates(instance, tol=tol))
    candidate_generation_time_seconds = 0.0

    if cfg.theta_order != "increasing":
        return _global_result(
            instance=instance,
            status="error",
            objective_value=float("-inf"),
            selected_options=None,
            selected_theta=None,
            lower_bound=float("-inf"),
            upper_bound=float("inf"),
            start_time=start,
            config=cfg,
            records=[],
            total_nodes=0,
            total_root_lp_time=0.0,
            total_bnb_time=0.0,
            diagnostics={"error": "parametric sweep currently requires theta_order='increasing'"},
            message="parametric sweep currently requires theta_order='increasing'",
        )

    incumbent_selection: Optional[List[int]] = None
    incumbent_value = float("-inf")
    incumbent_theta: Optional[float] = None
    if cfg.use_hullround_incumbent:
        try:
            hr = solve_hullround(instance, upgrade_completion=True)
            if hr.is_feasible and hr.selections and compute_certificate(instance, hr.selections) >= -tol:
                incumbent_selection = list(map(int, hr.selections))
                incumbent_value = float(hr.objective)
                incumbent_theta = float(hr.theta)
        except Exception:
            pass

    interruption_status: Optional[str] = None
    message = ""
    state_iter = iter_parametric_theta_states(instance, tol=tol, config=sw_cfg)
    solver_sweep_records: List[Dict[str, object]] = []

    for state_idx, state in enumerate(state_iter):
        theta = float(state.theta)
        solver_sweep_records.append(dict(state.record.__dict__))
        elapsed = time.perf_counter() - start
        if cfg.time_limit_seconds is not None and elapsed >= cfg.time_limit_seconds:
            interruption_status = "time_limit"
            message = "global time limit reached"
            theta_upper_bounds.append(float(state.lp_bound.lp_upper_bound))
            if state_idx + 1 < candidate_count_total:
                theta_upper_bounds.append(float("inf"))
            records.append(
                GlobalThetaRecord(
                    theta=theta,
                    status="not_run_time_limit",
                    fixed_theta_capacity=state.lp_bound.capacity,
                    fixed_theta_lp_upper_bound=state.lp_bound.lp_upper_bound,
                    incumbent_before_theta=incumbent_value,
                    bnb_objective=float("-inf"),
                    bnb_upper_bound=state.lp_bound.lp_upper_bound,
                    bnb_gap=float("inf"),
                    nodes_explored=0,
                    runtime_seconds=0.0,
                    pruned_by_bound=False,
                    infeasible_capacity=state.lp_bound.infeasible_capacity,
                    root_lp_status=state.lp_bound.root_lp_status,
                    incumbent_after_theta=incumbent_value,
                    root_lp_runtime_seconds=state.lp_bound.runtime_seconds,
                    diagnostics={"sweep": state.record.__dict__} if cfg.collect_diagnostics else {},
                )
            )
            break
        if cfg.node_limit is not None and total_nodes >= cfg.node_limit:
            interruption_status = "node_limit"
            message = "global node limit reached"
            theta_upper_bounds.append(float(state.lp_bound.lp_upper_bound))
            if state_idx + 1 < candidate_count_total:
                theta_upper_bounds.append(float("inf"))
            records.append(
                GlobalThetaRecord(
                    theta=theta,
                    status="not_run_node_limit",
                    fixed_theta_capacity=state.lp_bound.capacity,
                    fixed_theta_lp_upper_bound=state.lp_bound.lp_upper_bound,
                    incumbent_before_theta=incumbent_value,
                    bnb_objective=float("-inf"),
                    bnb_upper_bound=state.lp_bound.lp_upper_bound,
                    bnb_gap=float("inf"),
                    nodes_explored=0,
                    runtime_seconds=0.0,
                    pruned_by_bound=False,
                    infeasible_capacity=state.lp_bound.infeasible_capacity,
                    root_lp_status=state.lp_bound.root_lp_status,
                    incumbent_after_theta=incumbent_value,
                    root_lp_runtime_seconds=state.lp_bound.runtime_seconds,
                    diagnostics={"sweep": state.record.__dict__} if cfg.collect_diagnostics else {},
                )
            )
            break

        incumbent_before = incumbent_value
        lp_bound = state.lp_bound
        total_root_lp_time += float(lp_bound.runtime_seconds)
        if lp_bound.infeasible_capacity or not lp_bound.lp_feasible:
            theta_upper_bounds.append(float("-inf"))
            records.append(
                GlobalThetaRecord(
                    theta=theta,
                    status="infeasible_capacity" if lp_bound.infeasible_capacity else "infeasible",
                    fixed_theta_capacity=lp_bound.capacity,
                    fixed_theta_lp_upper_bound=lp_bound.lp_upper_bound,
                    incumbent_before_theta=incumbent_before,
                    bnb_objective=float("-inf"),
                    bnb_upper_bound=float("-inf"),
                    bnb_gap=float("inf"),
                    nodes_explored=0,
                    runtime_seconds=0.0,
                    pruned_by_bound=False,
                    infeasible_capacity=lp_bound.infeasible_capacity,
                    root_lp_status=lp_bound.root_lp_status,
                    incumbent_after_theta=incumbent_value,
                    root_lp_runtime_seconds=lp_bound.runtime_seconds,
                    diagnostics={"sweep": state.record.__dict__} if cfg.collect_diagnostics else {},
                )
            )
            continue

        if lp_bound.lp_upper_bound <= incumbent_value + tol:
            cert_time = 0.0
            cert_count = 0
            incumbent_cert_ok = False
            if incumbent_selection is not None:
                cert_start = time.perf_counter()
                incumbent_cert_ok = compute_certificate(instance, incumbent_selection) >= -tol
                cert_time = time.perf_counter() - cert_start
                cert_count = 1
            pruned_diag = {"sweep": state.record.__dict__} if cfg.collect_diagnostics else {}
            if cfg.collect_diagnostics:
                pruned_diag = dict(pruned_diag)
                pruned_diag["certificate_validation_time_seconds"] = float(cert_time)
                pruned_diag["certificate_validation_count"] = int(cert_count)
            theta_upper_bounds.append(lp_bound.lp_upper_bound)
            records.append(
                GlobalThetaRecord(
                    theta=theta,
                    status="pruned_bound",
                    fixed_theta_capacity=lp_bound.capacity,
                    fixed_theta_lp_upper_bound=lp_bound.lp_upper_bound,
                    incumbent_before_theta=incumbent_before,
                    bnb_objective=float("-inf"),
                    bnb_upper_bound=lp_bound.lp_upper_bound,
                    bnb_gap=0.0,
                    nodes_explored=0,
                    runtime_seconds=0.0,
                    pruned_by_bound=True,
                    infeasible_capacity=False,
                    root_lp_status=lp_bound.root_lp_status,
                    incumbent_after_theta=incumbent_value,
                    root_lp_runtime_seconds=lp_bound.runtime_seconds,
                    robust_certificate_passed=incumbent_cert_ok,
                    diagnostics=pruned_diag,
                )
            )
            continue

        fixed_time: Optional[float] = cfg.fixed_theta_time_limit_seconds
        if cfg.time_limit_seconds is not None:
            remaining = max(0.0, cfg.time_limit_seconds - (time.perf_counter() - start))
            fixed_time = remaining if fixed_time is None else min(fixed_time, remaining)
        fixed_nodes: Optional[int] = cfg.fixed_theta_node_limit
        if cfg.node_limit is not None:
            remaining_nodes = max(0, cfg.node_limit - total_nodes)
            fixed_nodes = remaining_nodes if fixed_nodes is None else min(fixed_nodes, remaining_nodes)

        initial_selection = incumbent_selection if incumbent_selection is not None else None
        initial_feasible_for_theta = False
        if initial_selection is not None:
            initial_flags = validate_fixed_theta_selection(
                state.cache.data.values,
                state.cache.data.costs,
                state.cache.data.capacity,
                initial_selection,
                tol,
            )
            initial_feasible_for_theta = bool(
                initial_flags["valid_length"] and initial_flags["valid_indices"] and initial_flags["capacity_feasible"]
            )

        bnb_start = time.perf_counter()
        bnb = solve_fixed_theta_bnb(
            instance,
            theta,
            FixedThetaBNBConfig(
                tolerance=tol,
                time_limit_seconds=fixed_time,
                node_limit=fixed_nodes,
                use_greedy_incumbent=cfg.use_fixed_theta_greedy_incumbent,
                objective_cutoff=incumbent_value if cfg.use_objective_cutoff and math.isfinite(incumbent_value) else None,
                initial_incumbent_selection=initial_selection if initial_feasible_for_theta else None,
                initial_incumbent_value=incumbent_value if initial_feasible_for_theta and math.isfinite(incumbent_value) else None,
                use_cutoff_pruning=cfg.use_objective_cutoff,
                use_cache=cfg.use_caches,
                branching_rule=cfg.branching_rule,
                child_ordering=cfg.child_ordering,
                strong_branching_candidates=cfg.strong_branching_candidates,
                strong_branching_depth_limit=cfg.strong_branching_depth_limit,
                strong_branching_max_children=cfg.strong_branching_max_children,
                strong_branching_use_fast_bound=cfg.strong_branching_use_fast_bound,
                incumbent_heuristic=cfg.incumbent_heuristic,
                use_local_incumbent_improvement=cfg.use_local_incumbent_improvement,
                local_search_max_passes=cfg.local_search_max_passes,
                local_search_max_swaps=cfg.local_search_max_swaps,
                local_search_neighborhood=cfg.local_search_neighborhood,
                local_search_max_pair_evaluations=cfg.local_search_max_pair_evaluations,
                collect_diagnostics=cfg.collect_diagnostics,
                max_diagnostic_nodes=cfg.max_diagnostic_nodes,
                diagnostic_sample_rate=cfg.diagnostic_sample_rate,
                profile_timing=cfg.profile_timing,
                use_fast_residual_lp_bound=cfg.use_fast_residual_lp_bound,
                use_cheap_prebound=cfg.use_cheap_prebound,
                use_min_cost_infeasibility_check=cfg.use_min_cost_infeasibility_check,
                use_bound_cache=cfg.use_bound_cache,
                bound_cache_max_entries=cfg.bound_cache_max_entries,
            ),
            fixed_theta_cache=state.cache,
        )
        total_bnb_time += time.perf_counter() - bnb_start
        total_nodes += int(bnb.nodes_explored)
        theta_upper_bounds.append(bnb.upper_bound if math.isfinite(bnb.upper_bound) else lp_bound.lp_upper_bound)

        if bnb.status in {"time_limit", "node_limit"} and interruption_status is None:
            interruption_status = bnb.status
            message = bnb.message
            if state_idx + 1 < candidate_count_total:
                theta_upper_bounds.append(float("inf"))
        if bnb.status == "error" and interruption_status is None:
            interruption_status = "error"
            message = bnb.message
            if state_idx + 1 < candidate_count_total:
                theta_upper_bounds.append(float("inf"))

        robust_passed = False
        cert_time = 0.0
        cert_count = 0
        if bnb.selected_options is not None and bnb.objective_value > incumbent_value + tol:
            cert_start = time.perf_counter()
            cert = compute_certificate(instance, bnb.selected_options)
            cert_time += time.perf_counter() - cert_start
            cert_count += 1
            if cert >= -tol:
                incumbent_selection = list(map(int, bnb.selected_options))
                incumbent_value = float(bnb.objective_value)
                incumbent_theta = theta
                robust_passed = True
        elif bnb.selected_options is not None:
            cert_start = time.perf_counter()
            robust_passed = compute_certificate(instance, bnb.selected_options) >= -tol
            cert_time += time.perf_counter() - cert_start
            cert_count += 1

        diag = bnb.diagnostics if cfg.collect_diagnostics or cfg.profile_timing else {}
        if cfg.collect_diagnostics or cfg.profile_timing:
            diag = dict(diag)
            diag["sweep"] = state.record.__dict__
            diag["certificate_validation_time_seconds"] = float(cert_time)
            diag["certificate_validation_count"] = int(cert_count)
        records.append(
            GlobalThetaRecord(
                theta=theta,
                status=bnb.status,
                fixed_theta_capacity=bnb.capacity,
                fixed_theta_lp_upper_bound=lp_bound.lp_upper_bound,
                incumbent_before_theta=incumbent_before,
                bnb_objective=bnb.objective_value,
                bnb_upper_bound=bnb.upper_bound,
                bnb_gap=bnb.absolute_gap,
                nodes_explored=bnb.nodes_explored,
                runtime_seconds=bnb.runtime_seconds,
                pruned_by_bound=False,
                infeasible_capacity=False,
                root_lp_status=lp_bound.root_lp_status,
                incumbent_after_theta=incumbent_value,
                bnb_lower_bound=bnb.lower_bound,
                nodes_pruned_bound=bnb.nodes_pruned_bound,
                nodes_pruned_infeasible=bnb.nodes_pruned_infeasible,
                cutoff_used=cfg.use_objective_cutoff and math.isfinite(incumbent_before),
                initial_incumbent_feasible_for_theta=initial_feasible_for_theta,
                robust_certificate_passed=robust_passed,
                root_lp_runtime_seconds=lp_bound.runtime_seconds,
                diagnostics=diag,
            )
        )

    upper_bound = max(theta_upper_bounds + ([incumbent_value] if math.isfinite(incumbent_value) else []), default=float("-inf"))
    lower_bound = incumbent_value if incumbent_selection is not None else float("-inf")
    if incumbent_selection is not None and upper_bound <= lower_bound + tol:
        status = "optimal"
    elif incumbent_selection is None and records and all(
        r.status in {"infeasible", "infeasible_capacity", "pruned_bound"} for r in records
    ):
        status = "infeasible"
    elif interruption_status in {"time_limit", "node_limit", "error"}:
        status = interruption_status
    else:
        unresolved = [r for r in records if r.status in {"time_limit", "node_limit", "error", "not_run_time_limit", "not_run_node_limit"}]
        if unresolved:
            status = "time_limit" if any("time" in r.status for r in unresolved) else "node_limit"
        else:
            status = "optimal" if incumbent_selection is not None else "infeasible"

    sweep_records = solver_sweep_records
    rebuilds = sum(int(r.get("hull_rebuilds", 0)) for r in sweep_records)
    reuses = sum(int(r.get("hull_reuses", 0)) for r in sweep_records)
    item_hulls_changed = sum(int(r.get("item_hulls_changed", 0)) for r in sweep_records)
    item_hulls_unchanged = sum(int(r.get("item_hulls_unchanged", 0)) for r in sweep_records)
    candidate_generation_time_seconds = float(
        sum(float((r.get("diagnostics") or {}).get("candidate_generation_time_seconds", 0.0)) for r in sweep_records)
    )
    validation_checks = sum(1 for r in sweep_records if bool(r.get("validation_checked", False)))
    certificate_validation_time = 0.0
    certificate_validation_count = 0
    for rec in records:
        diag = rec.diagnostics or {}
        if "certificate_validation_time_seconds" in diag:
            certificate_validation_time += float(diag.get("certificate_validation_time_seconds", 0.0) or 0.0)
            certificate_validation_count += int(diag.get("certificate_validation_count", 0) or 0)
    prune_count_by_reason = {
        "lp_bound": sum(1 for r in records if r.status == "pruned_bound"),
        "infeasible_capacity": sum(1 for r in records if r.status == "infeasible_capacity"),
        "infeasible_fixed_theta": sum(1 for r in records if r.status == "infeasible"),
    }
    validation_mode = "off"
    if sw_cfg.validate_against_recompute:
        validation_mode = "exhaustive" if sw_cfg.max_recompute_checks is None else "sampled"
    diagnostics.update(
        {
            "theta_count_total": len(records),
            "theta_count_candidate_total": int(candidate_count_total),
            "theta_candidate_source": "full_original_breakpoints",
            "theta_count_states_evaluated": int(len(sweep_records)),
            "theta_values": [float(r.theta) for r in records],
            "candidate_generation_time_seconds": candidate_generation_time_seconds,
            "sweep_update_time_seconds": float(sum(float(r.get("update_time_seconds", 0.0)) for r in sweep_records)),
            "baseline_update_time_seconds": float(sum(float(r.get("baseline_time_seconds", 0.0)) for r in sweep_records)),
            "hull_rebuilds_total": int(rebuilds),
            "hull_reuses_total": int(reuses),
            "item_hulls_changed_total": int(item_hulls_changed),
            "item_hulls_unchanged_total": int(item_hulls_unchanged),
            "dirty_item_fraction": float(item_hulls_changed / max(1, item_hulls_changed + item_hulls_unchanged)),
            "hull_reuse_rate": float(reuses / max(1, rebuilds + reuses)),
            "hull_rebuilds_by_theta": [int(r.get("hull_rebuilds", 0)) for r in sweep_records],
            "hull_reuses_by_theta": [int(r.get("hull_reuses", 0)) for r in sweep_records],
            "root_lp_time_seconds": float(total_root_lp_time),
            "fixed_theta_bnb_time_seconds": float(total_bnb_time),
            "total_nodes_explored": int(total_nodes),
            "max_abs_s_theta_recompute_error": float(max([0.0] + [float(r.get("max_abs_s_theta_recompute_error", 0.0)) for r in sweep_records])),
            "max_abs_cost_recompute_error": float(max([0.0] + [float(r.get("max_abs_cost_recompute_error", 0.0)) for r in sweep_records])),
            "max_abs_capacity_recompute_error": float(max([0.0] + [float(r.get("max_abs_capacity_recompute_error", 0.0)) for r in sweep_records])),
            "max_abs_root_lp_recompute_error": float(max([0.0] + [float(r.get("max_abs_root_lp_recompute_error", 0.0)) for r in sweep_records])),
            "validation_mode": validation_mode,
            "validation_checks_performed": int(validation_checks),
            "validation_max_recompute_checks": sw_cfg.max_recompute_checks,
            "prune_count_by_reason": prune_count_by_reason,
            "certificate_validation_time_seconds": float(certificate_validation_time),
            "certificate_validation_count": int(certificate_validation_count),
            "robust_certificate_value": float(compute_certificate(instance, incumbent_selection)) if incumbent_selection is not None else float("-inf"),
            "robust_certificate_passed": bool(incumbent_selection is not None and compute_certificate(instance, incumbent_selection) >= -tol),
        }
    )

    return _global_result(
        instance=instance,
        status=status,
        objective_value=incumbent_value,
        selected_options=incumbent_selection,
        selected_theta=incumbent_theta,
        lower_bound=lower_bound,
        upper_bound=upper_bound if incumbent_selection is not None or math.isfinite(upper_bound) else float("-inf"),
        start_time=start,
        config=cfg,
        records=records,
        total_nodes=total_nodes,
        total_root_lp_time=total_root_lp_time,
        total_bnb_time=total_bnb_time,
        diagnostics=diagnostics,
        message=message,
    )


__all__ = [
    "ParametricThetaRecord",
    "ParametricThetaState",
    "ParametricThetaSweepConfig",
    "ParametricThetaSweepResult",
    "build_parametric_theta_sweep",
    "iter_parametric_theta_states",
    "solve_global_theta_bnb_sweep",
]
