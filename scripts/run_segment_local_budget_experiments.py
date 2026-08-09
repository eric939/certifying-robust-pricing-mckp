#!/usr/bin/env python3
"""Run small exact segment-local Gamma-budget experiments."""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robust_mckp import GlobalThetaBNBConfig, Option, PricingInstance  # noqa: E402
from robust_mckp.certificate import compute_certificate  # noqa: E402
from robust_mckp.exact_bnb import solve_global_theta_bnb  # noqa: E402
from robust_mckp.local_budget import (  # noqa: E402
    SegmentLocalExactConfig,
    build_local_theta_candidates,
    robust_certificate_segment_local,
    solve_segment_local_exact,
)


CSV_NAME = "local_budget_results.csv"


def objective(instance: PricingInstance, selection: Sequence[int]) -> float:
    return float(sum(instance.items[i][int(j)].value for i, j in enumerate(selection)))


def brute_force_local(instance: PricingInstance, segments: Sequence[object], gammas: Mapping[object, int]) -> tuple[str, float, float]:
    best = None
    best_obj = float("-inf")
    for combo in itertools.product(*[range(len(group)) for group in instance.items]):
        cert = robust_certificate_segment_local(instance, combo, segments, gammas)
        if cert < -1e-9:
            continue
        obj = objective(instance, combo)
        if obj > best_obj + 1e-9:
            best = list(map(int, combo))
            best_obj = obj
    if best is None:
        return "infeasible", float("-inf"), float("-inf")
    return "optimal", best_obj, robust_certificate_segment_local(instance, best, segments, gammas)


def make_instance(n: int, m: int, gamma: int, seed: int) -> PricingInstance:
    rng = np.random.default_rng(seed + 313 * n + 997 * m)
    items = []
    for _i in range(n):
        group = []
        for j in range(m):
            group.append(
                Option(
                    value=float(rng.integers(0, 50) + j),
                    margin=float(rng.integers(-4, 12) - 0.2 * j),
                    uncertainty=float(rng.integers(0, 5)),
                )
            )
        group[0] = Option(value=float(rng.integers(0, 15)), margin=float(rng.integers(4, 12)), uncertainty=0.0)
        items.append(group)
    return PricingInstance(items=items, gamma=gamma, name=f"local_n{n}_m{m}_g{gamma}_s{seed}")


def segment_setup(n: int, mode: str) -> tuple[List[str], Dict[str, int]]:
    if mode == "one_segment":
        return ["all"] * n, {"all": max(0, int(math.floor(math.sqrt(n))))}
    labels = ["A", "B", "C"] if mode == "three_segment" else ["A", "B"]
    segments = [labels[i % len(labels)] for i in range(n)]
    gammas = {g: max(0, int(math.floor(math.sqrt(sum(1 for s in segments if s == g)))) - 1) for g in labels}
    return segments, gammas


def theta_product(instance: PricingInstance, segments: Sequence[object]) -> int:
    candidates = build_local_theta_candidates(instance, segments)
    product = 1
    for vals in candidates.values():
        product *= len(vals)
    return int(product)


def base_row(case: str, n: int, m: int, seed: int, method: str) -> Dict[str, object]:
    return {
        "case": case,
        "n": n,
        "m": m,
        "seed": seed,
        "method": method,
        "status": "not_run",
        "certified_optimal": False,
        "objective_value": float("nan"),
        "runtime_seconds": 0.0,
        "theta_vector_count_total": 0,
        "product_guard_triggered": False,
        "bruteforce_parity_passed": False,
        "one_segment_parity_passed": False,
        "robust_certificate": float("nan"),
        "message": "",
    }


def run_case(case: str, n: int, m: int, seed: int, guard: int) -> List[Dict[str, object]]:
    gamma = max(0, int(math.floor(math.sqrt(n))))
    instance = make_instance(n, m, gamma, seed)
    segments, gammas = segment_setup(n, case)
    rows: List[Dict[str, object]] = []
    product = theta_product(instance, segments)

    row = base_row(case, n, m, seed, "segment_local_exact")
    t0 = time.perf_counter()
    local = solve_segment_local_exact(instance, segments, gammas, SegmentLocalExactConfig(max_theta_vectors=guard))
    row.update(
        {
            "status": local.status,
            "certified_optimal": local.status == "optimal",
            "objective_value": local.objective_value,
            "runtime_seconds": time.perf_counter() - t0,
            "theta_vector_count_total": local.theta_vector_count_total,
            "product_guard_triggered": local.status == "too_many_theta_vectors",
            "robust_certificate": local.robust_certificate,
            "message": local.message,
        }
    )
    if product <= 200000:
        brute_status, brute_obj, _cert = brute_force_local(instance, segments, gammas)
        row["bruteforce_parity_passed"] = bool(
            brute_status == local.status == "optimal" and abs(float(local.objective_value) - brute_obj) <= 1e-8
        )
    rows.append(row)

    if case == "one_segment":
        global_row = base_row(case, n, m, seed, "global_exact")
        t0 = time.perf_counter()
        global_res = solve_global_theta_bnb(instance, GlobalThetaBNBConfig(use_hullround_incumbent=False))
        global_row.update(
            {
                "status": global_res.status,
                "certified_optimal": global_res.status == "optimal",
                "objective_value": global_res.objective_value,
                "runtime_seconds": time.perf_counter() - t0,
                "theta_vector_count_total": product,
                "one_segment_parity_passed": bool(
                    local.status == global_res.status == "optimal"
                    and abs(float(local.objective_value) - float(global_res.objective_value)) <= 1e-8
                ),
                "robust_certificate": global_res.robust_certificate,
            }
        )
        rows.append(global_row)

    guard_row = base_row("guard", n, m, seed, "segment_local_guard")
    guarded = solve_segment_local_exact(instance, segments, gammas, SegmentLocalExactConfig(max_theta_vectors=1))
    guard_row.update(
        {
            "status": guarded.status,
            "theta_vector_count_total": guarded.theta_vector_count_total,
            "product_guard_triggered": guarded.status == "too_many_theta_vectors",
            "message": guarded.message,
        }
    )
    rows.append(guard_row)
    return rows


def grid(args: argparse.Namespace) -> Iterable[tuple[str, int, int, int]]:
    if args.smoke:
        yield "one_segment", 6, 3, 0
        yield "two_segment", 6, 3, 1
        yield "three_segment", 6, 3, 2
        return
    ns = [30, 40, 50]
    ms = [8, 10]
    seeds = [0, 1]
    emitted = 0
    for case in ["one_segment", "two_segment", "three_segment"]:
        for n in ns:
            for m in ms:
                for seed in seeds:
                    if args.max_instances is not None and emitted >= args.max_instances:
                        return
                    emitted += 1
                    yield case, n, m, seed


def load_existing(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_key(row: Dict[str, object]) -> tuple[object, ...]:
    return row["case"], row["n"], row["m"], row["seed"], row["method"]


def write_rows(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> List[Dict[str, object]]:
    path = args.output_dir / CSV_NAME
    rows = load_existing(path) if args.resume else []
    done = {row_key(r) for r in rows}
    cases = list(grid(args))
    for idx, (case, n, m, seed) in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case} n={n} m={m} seed={seed}", flush=True)
        for row in run_case(case, n, m, seed, args.max_theta_vectors):
            if row_key(row) in done:
                continue
            rows.append(row)
            done.add(row_key(row))
            write_rows(rows, path)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--paper-lite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "segment_local_budget")
    parser.add_argument("--max-theta-vectors", type=int, default=100000)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)
    print(f"Wrote {args.output_dir / CSV_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
