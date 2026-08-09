#!/usr/bin/env python3
"""Run adversarial and oracle checks for Paper A's mathematical claims.

This is a verification driver, not a benchmark.  It compares the public
algorithms with exhaustive enumeration and an independent LP solver on small
instances, checks the finite-breakpoint identity selection by selection, and
exercises the guarded sweep against independent reconstruction.  Any mismatch
is retained in the generated JSON report and makes the command fail.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_mckp import GlobalThetaBNBConfig, Option, PricingInstance, solve  # noqa: E402
from robust_mckp.certificate import certificate_is_feasible, compute_certificate  # noqa: E402
from robust_mckp.exact_bnb import (  # noqa: E402
    brute_force_global_robust,
    build_fixed_theta_data,
    build_full_theta_candidates,
    compute_fixed_theta_lp_upper_bound,
    solve_global_theta_bnb,
)
from robust_mckp.parametric_sweep import (  # noqa: E402
    ParametricThetaSweepConfig,
    build_parametric_theta_sweep,
)


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def _instance(rng: np.random.Generator, *, floating: bool) -> PricingInstance:
    n = int(rng.integers(1, 7 if not floating else 6))
    m = int(rng.integers(1, 5))
    gamma = int(rng.integers(0, n + 1))
    items: List[List[Option]] = []
    for _i in range(n):
        group: List[Option] = []
        for j in range(m):
            if floating:
                scale = 10.0 ** int(rng.integers(-6, 7))
                value = float(rng.uniform(0.0, 30.0) * scale)
                margin = float(rng.uniform(-6.0, 15.0) * scale)
                uncertainty = float(rng.uniform(-5.0, 5.0) * scale)
                if rng.random() < 0.2:
                    uncertainty = float(1.0 + (j + 1) * 5e-11)
            else:
                value = float(rng.integers(0, 31))
                margin = float(rng.integers(-6, 15))
                uncertainty = float(rng.integers(-5, 6))
            group.append(Option(value=value, margin=margin, uncertainty=uncertainty))
        items.append(group)
    return PricingInstance(items=items, gamma=gamma)


def _selection_residual(instance: PricingInstance, selection: Sequence[int], theta: float) -> float:
    return float(
        sum(
            instance.items[i][int(j)].margin
            - max(0.0, abs(instance.items[i][int(j)].uncertainty) - float(theta))
            for i, j in enumerate(selection)
        )
        - instance.gamma * float(theta)
    )


def _lp_oracle(instance: PricingInstance, theta: float) -> tuple[str, float]:
    data = build_fixed_theta_data(instance, theta)
    if data.capacity < 0.0:
        return "infeasible", float("-inf")
    offsets: List[int] = []
    total = 0
    for group in data.values:
        offsets.append(total)
        total += int(group.size)
    objective = -np.concatenate(data.values)
    cap = np.concatenate(data.costs).reshape(1, -1)
    aeq = np.zeros((instance.n_items, total), dtype=float)
    for i, group in enumerate(data.values):
        aeq[i, offsets[i] : offsets[i] + group.size] = 1.0
    result = linprog(
        objective,
        A_ub=cap,
        b_ub=np.array([data.capacity]),
        A_eq=aeq,
        b_eq=np.ones(instance.n_items),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if result.status == 2:
        return "infeasible", float("-inf")
    if not result.success:
        raise RuntimeError(f"HiGHS LP oracle failed: {result.status} {result.message}")
    return "optimal", float(-result.fun)


def _close(a: float, b: float, *, atol: float = 1e-8, rtol: float = 1e-10) -> bool:
    if a == b:
        return True
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    failures: List[Dict[str, Any]] = []
    counts = {
        "global_integer_oracle_cases": 0,
        "global_floating_oracle_cases": 0,
        "selection_breakpoint_identity_checks": 0,
        "fixed_theta_lp_oracle_checks": 0,
        "sweep_reconstruction_cases": 0,
        "hullround_certificate_cases": 0,
    }

    def fail(kind: str, case: int, **details: Any) -> None:
        if len(failures) < args.max_failure_records:
            failures.append({"kind": kind, "case": case, **details})

    start = time.perf_counter()
    for floating, key in [(False, "global_integer_oracle_cases"), (True, "global_floating_oracle_cases")]:
        for case in range(args.cases):
            instance = _instance(rng, floating=floating)
            brute = brute_force_global_robust(instance, tol=0.0)
            exact = solve_global_theta_bnb(
                instance,
                GlobalThetaBNBConfig(
                    tolerance=0.0,
                    use_hullround_incumbent=False,
                    use_fast_residual_lp_bound=True,
                ),
            )
            counts[key] += 1
            if exact.status != brute.status or (
                brute.status == "optimal" and not _close(exact.objective_value, brute.objective_value)
            ):
                fail(
                    "global_vs_bruteforce",
                    case,
                    floating=floating,
                    gamma=instance.gamma,
                    group_sizes=[len(g) for g in instance.items],
                    brute_status=brute.status,
                    brute_objective=brute.objective_value,
                    exact_status=exact.status,
                    exact_objective=exact.objective_value,
                )

            candidates = build_full_theta_candidates(instance)
            selection_iter = itertools.product(*(range(len(g)) for g in instance.items))
            for selection_index, selection in enumerate(selection_iter):
                if selection_index >= args.max_selections_per_instance:
                    break
                direct = compute_certificate(instance, selection)
                envelope = max(_selection_residual(instance, selection, theta) for theta in candidates)
                counts["selection_breakpoint_identity_checks"] += 1
                if not _close(direct, envelope, atol=2e-8, rtol=2e-12):
                    fail(
                        "breakpoint_identity",
                        case,
                        floating=floating,
                        selection=list(selection),
                        direct_certificate=direct,
                        max_breakpoint_residual=envelope,
                    )

            if case < args.lp_cases:
                for theta in candidates[: args.max_thetas_per_instance]:
                    bound = compute_fixed_theta_lp_upper_bound(
                        instance, theta, tol=args.lp_feasibility_tolerance
                    )
                    oracle_status, oracle_value = _lp_oracle(instance, theta)
                    counts["fixed_theta_lp_oracle_checks"] += 1
                    if oracle_status == "optimal":
                        if not bound.lp_feasible or not _close(bound.lp_upper_bound, oracle_value, atol=2e-7):
                            fail(
                                "fixed_theta_lp_oracle",
                                case,
                                theta=theta,
                                bound_status=bound.root_lp_status,
                                bound_value=bound.lp_upper_bound,
                                oracle_value=oracle_value,
                            )
                    elif bound.lp_feasible:
                        fail(
                            "fixed_theta_lp_false_feasible",
                            case,
                            theta=theta,
                            bound_status=bound.root_lp_status,
                        )

            if case < args.sweep_cases:
                try:
                    sweep = build_parametric_theta_sweep(
                        instance,
                        config=ParametricThetaSweepConfig(
                            # Candidate identity and structural preprocessing remain
                            # exact.  This tolerance is used only when comparing two
                            # algebraically equivalent floating-point accumulation
                            # paths in the optional sweep reconstruction check.
                            tol=args.sweep_validation_tolerance,
                            validate_against_recompute=True,
                            reuse_hulls=True,
                        ),
                    )
                    counts["sweep_reconstruction_cases"] += 1
                    if len(sweep.states) != len(candidates):
                        fail(
                            "sweep_candidate_count",
                            case,
                            expected=len(candidates),
                            observed=len(sweep.states),
                        )
                except Exception as exc:
                    fail("sweep_reconstruction", case, error=str(exc))

            hullround = solve(instance, upgrade_completion=False)
            counts["hullround_certificate_cases"] += 1
            if hullround.is_feasible:
                diag = (hullround.metadata or {}).get("instrumentation", {})
                loss = float(diag.get("selected_theta_lp_minus_round_down", float("inf")))
                delta = float(diag.get("selected_theta_delta_v_max", float("-inf")))
                direct = compute_certificate(instance, hullround.selections)
                if not certificate_is_feasible(instance, hullround.selections) or loss > delta + 2e-7 * max(1.0, abs(delta), abs(loss)):
                    fail(
                        "hullround_certificate",
                        case,
                        floating=floating,
                        direct_certificate=direct,
                        lp_minus_round_down=loss,
                        delta_v_max=delta,
                    )

    # Permanent regression: every binary64-distinct threshold must survive.
    delta = 9e-11
    close_instance = PricingInstance(
        items=[
            [Option(value=1.0, margin=(1.0 + delta) / 100.0, uncertainty=1.0 if i == 0 else 1.0 + delta)]
            for i in range(100)
        ],
        gamma=1,
        name="unique_feasible_close_breakpoint",
    )
    close_candidates = build_full_theta_candidates(close_instance, tol=1e-9)
    close_result = solve_global_theta_bnb(
        close_instance,
        GlobalThetaBNBConfig(tolerance=1e-12, use_hullround_incumbent=False),
    )
    if close_candidates != [0.0, 1.0, 1.0 + delta] or close_result.selected_theta != 1.0 + delta:
        fail(
            "close_breakpoint_regression",
            -1,
            candidates=close_candidates,
            selected_theta=close_result.selected_theta,
            status=close_result.status,
        )

    return {
        "schema_version": 1,
        "purpose": "Paper A mathematical and implementation oracle audit",
        "protocol": {
            "seed": args.seed,
            "cases_per_numeric_family": args.cases,
            "lp_cases_per_numeric_family": args.lp_cases,
            "sweep_cases_per_numeric_family": args.sweep_cases,
            "max_selections_per_instance": args.max_selections_per_instance,
            "max_thetas_per_instance": args.max_thetas_per_instance,
            "branch_and_bound_comparison_tolerance": 0.0,
            "lp_feasibility_tolerance": args.lp_feasibility_tolerance,
            "sweep_reconstruction_absolute_tolerance": args.sweep_validation_tolerance,
            "structural_threshold_clustering": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy_version,
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _git(["status", "--porcelain"]),
        },
        "counts": counts,
        "close_breakpoint_regression": {
            "candidates": close_candidates,
            "result_status": close_result.status,
            "selected_theta": close_result.selected_theta,
        },
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
        "elapsed_seconds": time.perf_counter() - start,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--lp-cases", type=int, default=50)
    parser.add_argument("--sweep-cases", type=int, default=30)
    parser.add_argument("--max-selections-per-instance", type=int, default=256)
    parser.add_argument("--max-thetas-per-instance", type=int, default=20)
    parser.add_argument("--sweep-validation-tolerance", type=float, default=1e-9)
    parser.add_argument("--lp-feasibility-tolerance", type=float, default=1e-12)
    parser.add_argument("--max-failure-records", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "failure_count": report["failure_count"], "counts": report["counts"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
