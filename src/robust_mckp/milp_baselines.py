"""Optional MILP baselines for theta-decomposed robust MCKP."""
from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .certificate import certificate_is_feasible, compute_certificate
from .exact_bnb import build_fixed_theta_data, build_full_theta_candidates, cost_for_selection
from .model import PricingInstance


@dataclass
class MILPBaselineResult:
    """Standard result object for optional theta-decomposition MILP baselines."""

    backend: str
    available: bool
    status: str
    objective_value: float
    selected_options: Optional[List[int]]
    selected_theta: Optional[float]
    robust_certificate: float
    runtime_seconds: float
    theta_count_total: int
    theta_count_solved: int
    theta_count_skipped: int
    gap: float
    message: str = ""


def _objective(instance: PricingInstance, selections: Sequence[int]) -> float:
    return float(sum(instance.items[i][int(j)].value for i, j in enumerate(selections)))


def _not_available(backend: str, message: str) -> MILPBaselineResult:
    return MILPBaselineResult(
        backend=backend,
        available=False,
        status="not_available",
        objective_value=float("-inf"),
        selected_options=None,
        selected_theta=None,
        robust_certificate=float("-inf"),
        runtime_seconds=0.0,
        theta_count_total=0,
        theta_count_solved=0,
        theta_count_skipped=0,
        gap=float("inf"),
        message=message,
    )


def _solve_fixed_theta_scipy(instance: PricingInstance, theta: float, time_limit: Optional[float]) -> tuple[str, Optional[List[int]], float, float]:
    try:
        import scipy.optimize as opt  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        return "not_available", None, float("-inf"), float("inf")
    if not hasattr(opt, "milp"):
        return "not_available", None, float("-inf"), float("inf")
    data = build_fixed_theta_data(instance, theta)
    if data.capacity < -1e-9:
        return "infeasible_capacity", None, float("-inf"), 0.0
    sizes = [len(v) for v in data.values]
    total = sum(sizes)
    c = -np.concatenate(data.values)
    integrality = np.ones(total, dtype=int)
    bounds = opt.Bounds(lb=np.zeros(total), ub=np.ones(total))
    a_eq = np.zeros((len(sizes), total))
    offset = 0
    for i, size in enumerate(sizes):
        a_eq[i, offset : offset + size] = 1.0
        offset += size
    constraints = [
        opt.LinearConstraint(a_eq, np.ones(len(sizes)), np.ones(len(sizes))),
        opt.LinearConstraint(np.concatenate(data.costs)[None, :], -np.inf, np.array([data.capacity])),
    ]
    options = {"time_limit": float(time_limit)} if time_limit is not None else None
    res = opt.milp(c, integrality=integrality, bounds=bounds, constraints=constraints, options=options)
    status_code = int(getattr(res, "status", -1))
    if status_code != 0 or getattr(res, "x", None) is None:
        if status_code == 2:
            return "infeasible", None, float("-inf"), float(getattr(res, "mip_gap", float("inf")))
        return "limited", None, float("-inf"), float(getattr(res, "mip_gap", float("inf")))
    x = np.array(res.x, dtype=float)
    selections: List[int] = []
    offset = 0
    for size in sizes:
        selections.append(int(np.argmax(x[offset : offset + size])))
        offset += size
    if cost_for_selection(data.costs, selections) > data.capacity + 1e-7:
        return "infeasible", None, float("-inf"), float(getattr(res, "mip_gap", float("inf")))
    return "optimal", selections, float(-res.fun), float(getattr(res, "mip_gap", 0.0))


def _solve_fixed_theta_scip(instance: PricingInstance, theta: float, time_limit: Optional[float]) -> tuple[str, Optional[List[int]], float, float]:
    try:
        from pyscipopt import Model, quicksum  # type: ignore
    except Exception:  # pragma: no cover - depends on environment
        return "not_available", None, float("-inf"), float("inf")
    data = build_fixed_theta_data(instance, theta)
    if data.capacity < -1e-9:
        return "infeasible_capacity", None, float("-inf"), 0.0
    model = Model("fixed_theta_mckp")
    model.hideOutput()
    if time_limit is not None:
        model.setRealParam("limits/time", float(time_limit))
    z = {}
    for i, vals in enumerate(data.values):
        for j in range(len(vals)):
            z[i, j] = model.addVar(vtype="B", name=f"z_{i}_{j}")
    for i, vals in enumerate(data.values):
        model.addCons(quicksum(z[i, j] for j in range(len(vals))) == 1.0)
    model.addCons(
        quicksum(float(data.costs[i][j]) * z[i, j] for i, vals in enumerate(data.values) for j in range(len(vals)))
        <= float(data.capacity)
    )
    model.setObjective(
        quicksum(float(data.values[i][j]) * z[i, j] for i, vals in enumerate(data.values) for j in range(len(vals))),
        "maximize",
    )
    model.optimize()
    status = str(model.getStatus()).lower()
    if model.getNSols() <= 0:
        return ("infeasible" if status == "infeasible" else "limited"), None, float("-inf"), float("inf")
    selections = [max(range(len(vals)), key=lambda j: float(model.getVal(z[i, j]))) for i, vals in enumerate(data.values)]
    obj = float(sum(float(data.values[i][j]) for i, j in enumerate(selections)))
    try:
        gap = float(model.getGap())
    except Exception:
        gap = 0.0 if status == "optimal" else float("inf")
    return ("optimal" if status == "optimal" else "limited"), selections, obj, gap


def solve_theta_decomposition_milp_baseline(
    instance: PricingInstance,
    backend: str = "scipy_highs",
    *,
    time_limit_per_theta: Optional[float] = None,
    tol: float = 1e-9,
) -> MILPBaselineResult:
    """Run an optional MILP baseline over the full theta decomposition."""

    backend_key = backend.lower()
    if backend_key in {"scipy", "scipy_highs", "highs"}:
        if importlib.util.find_spec("scipy.optimize") is None and importlib.util.find_spec("scipy") is None:
            return _not_available(backend_key, "SciPy/HiGHS is not installed")
        solver = _solve_fixed_theta_scipy
    elif backend_key == "scip":
        if importlib.util.find_spec("pyscipopt") is None:
            return _not_available(backend_key, "PySCIPOpt/SCIP is not installed")
        solver = _solve_fixed_theta_scip
    elif backend_key == "gurobi":
        if importlib.util.find_spec("gurobipy") is None:
            return _not_available(backend_key, "gurobipy is not installed or licensed")
        return _not_available(backend_key, "Gurobi wrapper is not enabled in this lightweight baseline module")
    elif backend_key == "cplex":
        if importlib.util.find_spec("cplex") is None and importlib.util.find_spec("docplex") is None:
            return _not_available(backend_key, "CPLEX/docplex is not installed or licensed")
        return _not_available(backend_key, "CPLEX wrapper is not enabled in this lightweight baseline module")
    else:
        return _not_available(backend_key, f"unknown backend: {backend}")

    start = time.perf_counter()
    candidates = build_full_theta_candidates(instance, tol=tol)
    best_selection: Optional[List[int]] = None
    best_obj = float("-inf")
    best_theta: Optional[float] = None
    theta_solved = 0
    theta_skipped = 0
    saw_limited = False
    for theta in candidates:
        status, selections, obj, _gap = solver(instance, float(theta), time_limit_per_theta)
        if status == "not_available":
            return _not_available(backend_key, "SciPy/HiGHS is unavailable")
        if status in {"infeasible_capacity", "infeasible"}:
            theta_skipped += 1
            continue
        if status != "optimal":
            saw_limited = True
            theta_skipped += 1
            continue
        theta_solved += 1
        if selections is not None:
            cert = compute_certificate(instance, selections)
            robust_obj = _objective(instance, selections)
            if certificate_is_feasible(instance, selections) and robust_obj > best_obj + tol:
                best_selection = list(map(int, selections))
                best_obj = float(robust_obj)
                best_theta = float(theta)

    cert = compute_certificate(instance, best_selection) if best_selection is not None else float("-inf")
    if best_selection is not None and not saw_limited:
        status = "optimal"
        gap = 0.0
    elif best_selection is not None:
        status = "limited"
        gap = float("inf")
    else:
        status = "infeasible" if theta_solved == 0 and not saw_limited else "limited"
        gap = float("inf")
    return MILPBaselineResult(
        backend=backend_key,
        available=True,
        status=status,
        objective_value=float(best_obj) if best_selection is not None else float("-inf"),
        selected_options=best_selection,
        selected_theta=best_theta,
        robust_certificate=float(cert),
        runtime_seconds=float(time.perf_counter() - start),
        theta_count_total=len(candidates),
        theta_count_solved=int(theta_solved),
        theta_count_skipped=int(theta_skipped),
        gap=gap,
        message="theta-decomposition fixed-theta MILP baseline",
    )


__all__ = ["MILPBaselineResult", "solve_theta_decomposition_milp_baseline"]
