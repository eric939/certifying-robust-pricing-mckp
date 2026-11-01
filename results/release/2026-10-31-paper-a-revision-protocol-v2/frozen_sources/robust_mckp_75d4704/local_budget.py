"""Exact segment-local Gamma-budget extension for small theta products."""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .exact_bnb import (
    FixedThetaBNBConfig,
    FixedThetaData,
    _build_fixed_theta_cache,
    cost_for_selection,
    objective_for_selection,
    solve_fixed_theta_bnb,
)
from .model import PricingInstance
from .utils import top_gamma


@dataclass(frozen=True)
class SegmentLocalExactConfig:
    """Configuration for exact local-budget theta-vector enumeration."""

    tolerance: float = 1e-9
    max_theta_vectors: int = 100_000
    fixed_theta_time_limit_seconds: Optional[float] = None
    fixed_theta_node_limit: Optional[int] = None
    use_caches: bool = True
    use_fixed_theta_greedy_incumbent: bool = True


@dataclass
class SegmentLocalExactResult:
    """Result for exact segment-local robust solving."""

    status: str
    objective_value: float
    selected_options: Optional[List[int]]
    selected_theta_vector: Optional[Dict[object, float]]
    robust_certificate: float
    lower_bound: float
    upper_bound: float
    absolute_gap: float
    relative_gap: float
    theta_vector_count_total: int
    theta_vector_count_solved: int
    theta_vector_count_pruned: int
    theta_vector_count_infeasible: int
    total_nodes_explored: int
    runtime_seconds: float
    diagnostics: Dict[str, object] = field(default_factory=dict)
    message: str = ""


def _validate_segments(
    instance: PricingInstance,
    segments: Sequence[object],
    segment_gammas: Mapping[object, int],
) -> List[object]:
    if len(segments) != instance.n_items:
        raise ValueError("segments length must equal number of items")
    labels = list(dict.fromkeys(segments))
    missing = [g for g in labels if g not in segment_gammas]
    if missing:
        raise ValueError(f"missing segment Gamma values for segments: {missing}")
    counts = {g: sum(1 for seg in segments if seg == g) for g in labels}
    for g in labels:
        gamma = int(segment_gammas[g])
        if gamma < 0 or gamma > counts[g]:
            raise ValueError(f"segment Gamma for {g!r} must be in [0, number of items in segment]")
    return labels


def robust_certificate_segment_local(
    instance: PricingInstance,
    selection: Sequence[int],
    segments: Sequence[object],
    segment_gammas: Mapping[object, int],
) -> float:
    """Compute local-budget certificate for a selected option vector."""

    labels = _validate_segments(instance, segments, segment_gammas)
    if len(selection) != instance.n_items:
        raise ValueError("selection length must match number of items")
    margin_sum = 0.0
    by_segment: Dict[object, List[float]] = {g: [] for g in labels}
    for i, j in enumerate(selection):
        jj = int(j)
        if jj < 0 or jj >= len(instance.items[i]):
            raise ValueError(f"invalid option index {jj} for item {i}")
        opt = instance.items[i][jj]
        margin_sum += float(opt.margin)
        by_segment[segments[i]].append(abs(float(opt.uncertainty)))
    penalty = sum(top_gamma(np.array(vals, dtype=float), int(segment_gammas[g])) for g, vals in by_segment.items())
    return float(margin_sum - penalty)


def build_local_theta_candidates(
    instance: PricingInstance,
    segments: Sequence[object],
    tol: Optional[float] = None,
) -> Dict[object, List[float]]:
    """Build binary64-distinct segment-wise full breakpoint sets ``B_g``."""

    del tol

    labels = list(dict.fromkeys(segments))
    if len(segments) != instance.n_items:
        raise ValueError("segments length must equal number of items")
    candidates: Dict[object, List[float]] = {g: [0.0] for g in labels}
    for i, group in enumerate(instance.items):
        g = segments[i]
        candidates[g].extend(abs(float(opt.uncertainty)) for opt in group)
    return {g: sorted({float(value) for value in vals}) for g, vals in candidates.items()}


def _build_local_fixed_theta_data(
    instance: PricingInstance,
    segments: Sequence[object],
    segment_gammas: Mapping[object, int],
    theta_vector: Mapping[object, float],
) -> FixedThetaData:
    values: List[np.ndarray] = []
    s_theta_values: List[np.ndarray] = []
    costs: List[np.ndarray] = []
    baseline_indices: List[int] = []
    s_star_sum = 0.0
    for i, group in enumerate(instance.items):
        theta = float(theta_vector[segments[i]])
        v = np.array([opt.value for opt in group], dtype=float)
        s = np.array([opt.margin for opt in group], dtype=float)
        abs_t = np.array([abs(opt.uncertainty) for opt in group], dtype=float)
        st = s - np.maximum(0.0, abs_t - theta)
        baseline_idx = int(np.argmax(st))
        s_star = float(st[baseline_idx])
        values.append(v)
        s_theta_values.append(st)
        costs.append(np.maximum(s_star - st, 0.0))
        baseline_indices.append(baseline_idx)
        s_star_sum += s_star
    capacity = s_star_sum - sum(float(segment_gammas[g]) * float(theta_vector[g]) for g in theta_vector)
    return FixedThetaData(
        values=values,
        s_theta=s_theta_values,
        costs=costs,
        capacity=float(capacity),
        s_star_sum=float(s_star_sum),
        baseline_indices=baseline_indices,
    )


def _relative_gap(upper: float, lower: float, tol: float) -> float:
    if not math.isfinite(upper) or not math.isfinite(lower):
        return float("inf")
    gap = max(0.0, upper - lower)
    return 0.0 if gap <= tol else float(gap / max(abs(upper), abs(lower), tol))


def solve_segment_local_exact(
    instance: PricingInstance,
    segments: Sequence[object],
    segment_gammas: Mapping[object, int],
    config: Optional[SegmentLocalExactConfig] = None,
) -> SegmentLocalExactResult:
    """Solve segment-local robust MCKP exactly for small theta-vector products."""

    cfg = config or SegmentLocalExactConfig()
    tol = float(cfg.tolerance)
    start = time.perf_counter()
    labels = _validate_segments(instance, segments, segment_gammas)
    candidates = build_local_theta_candidates(instance, segments, tol)
    vector_count = 1
    for g in labels:
        vector_count *= len(candidates[g])
    if vector_count > int(cfg.max_theta_vectors):
        return SegmentLocalExactResult(
            status="too_many_theta_vectors",
            objective_value=float("-inf"),
            selected_options=None,
            selected_theta_vector=None,
            robust_certificate=float("-inf"),
            lower_bound=float("-inf"),
            upper_bound=float("inf"),
            absolute_gap=float("inf"),
            relative_gap=float("inf"),
            theta_vector_count_total=int(vector_count),
            theta_vector_count_solved=0,
            theta_vector_count_pruned=0,
            theta_vector_count_infeasible=0,
            total_nodes_explored=0,
            runtime_seconds=float(time.perf_counter() - start),
            diagnostics={"segment_labels": labels, "theta_counts_by_segment": {g: len(candidates[g]) for g in labels}},
            message="theta-vector product exceeds max_theta_vectors; no approximation was run",
        )

    best_selection: Optional[List[int]] = None
    best_obj = float("-inf")
    best_theta: Optional[Dict[object, float]] = None
    upper_bounds: List[float] = []
    solved = 0
    pruned = 0
    infeasible = 0
    total_nodes = 0

    for theta_values in itertools.product(*[candidates[g] for g in labels]):
        theta_vector = {g: float(theta_values[k]) for k, g in enumerate(labels)}
        data = _build_local_fixed_theta_data(instance, segments, segment_gammas, theta_vector)
        if data.capacity < -tol:
            infeasible += 1
            upper_bounds.append(float("-inf"))
            continue
        cache = _build_fixed_theta_cache(instance, 0.0, tol, data=data)
        # Cheap root-level prune using the incumbent and exact LP cache through B&B.
        result = solve_fixed_theta_bnb(
            instance,
            0.0,
            FixedThetaBNBConfig(
                tolerance=tol,
                time_limit_seconds=cfg.fixed_theta_time_limit_seconds,
                node_limit=cfg.fixed_theta_node_limit,
                use_greedy_incumbent=cfg.use_fixed_theta_greedy_incumbent,
                objective_cutoff=best_obj if math.isfinite(best_obj) else None,
                initial_incumbent_selection=best_selection,
                initial_incumbent_value=best_obj if best_selection is not None and math.isfinite(best_obj) else None,
                use_cutoff_pruning=True,
                use_cache=cfg.use_caches,
            ),
            fixed_theta_cache=cache,
        )
        total_nodes += int(result.nodes_explored)
        upper_bounds.append(result.upper_bound)
        if result.status == "infeasible":
            infeasible += 1
            continue
        if result.status == "cutoff_pruned":
            pruned += 1
            continue
        solved += 1 if result.status == "optimal" else 0
        if result.selected_options is not None:
            cert = robust_certificate_segment_local(instance, result.selected_options, segments, segment_gammas)
            if cert >= 0.0 and result.objective_value > best_obj + tol:
                best_selection = list(map(int, result.selected_options))
                best_obj = float(result.objective_value)
                best_theta = theta_vector

    lower = best_obj if best_selection is not None else float("-inf")
    upper = max(upper_bounds + ([lower] if math.isfinite(lower) else []), default=float("-inf"))
    cert = (
        robust_certificate_segment_local(instance, best_selection, segments, segment_gammas)
        if best_selection is not None
        else float("-inf")
    )
    if best_selection is not None and upper <= lower + tol:
        status = "optimal"
    elif best_selection is None and infeasible == vector_count:
        status = "infeasible"
    else:
        status = "limited"
    abs_gap = max(0.0, upper - lower) if math.isfinite(upper) and math.isfinite(lower) else float("inf")
    return SegmentLocalExactResult(
        status=status,
        objective_value=float(best_obj) if best_selection is not None else float("-inf"),
        selected_options=best_selection,
        selected_theta_vector=best_theta,
        robust_certificate=float(cert),
        lower_bound=float(lower),
        upper_bound=float(upper),
        absolute_gap=float(abs_gap),
        relative_gap=_relative_gap(upper, lower, tol),
        theta_vector_count_total=int(vector_count),
        theta_vector_count_solved=int(solved),
        theta_vector_count_pruned=int(pruned),
        theta_vector_count_infeasible=int(infeasible),
        total_nodes_explored=int(total_nodes),
        runtime_seconds=float(time.perf_counter() - start),
        diagnostics={
            "segment_labels": labels,
            "theta_counts_by_segment": {g: len(candidates[g]) for g in labels},
        },
    )


__all__ = [
    "SegmentLocalExactConfig",
    "SegmentLocalExactResult",
    "build_local_theta_candidates",
    "robust_certificate_segment_local",
    "solve_segment_local_exact",
]
