#!/usr/bin/env python3
"""Generate extended publishable evidence for Sections 6--7.

The baseline figures/tables are produced by ``run_publishable_experiments.py``
and ``run_solver_benchmarks.py``.  This script adds higher-cost diagnostic
experiments requested for the final paper audit: component ablations,
time-limited solver behavior, sensitivity grids, theta-loop decomposition,
out-of-model stress tests, all-Gamma retail decision paths, and tight bound
examples.  Outputs are written under ``results/publishable`` and are intended
to be consumed directly by the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in [ROOT, ROOT / "src", ROOT / "experiments_nested", ROOT / "experiments_case_retail"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments_nested._common import (  # noqa: E402
    build_prefix_instance,
    compute_hulls_for_theta,
    make_master_portfolio,
    round_down_value,
    selected_option_data,
)
from robust_mckp import PricingInstance, solve  # noqa: E402
from robust_mckp.certificate import compute_certificate  # noqa: E402
from robust_mckp.greedy import greedy_lp  # noqa: E402
from robust_mckp.rounding import round_lp_solution  # noqa: E402
from robust_mckp.utils import EPS  # noqa: E402
from scripts.run_publishable_experiments import (  # noqa: E402
    CSV_DIR,
    FIG_DIR,
    OUT,
    TABLE_DIR,
    apply_plot_style,
    fmt,
    fmt_pct,
    gamma_regime,
    hullround_metrics,
    median,
    option_arrays,
    q,
    read_csv,
    robust_residual_for_budget,
    simulate_iid_prefix,
    selected_prefix_arrays,
    write_csv,
)
from scripts.run_solver_benchmarks import solve_full_robust_scip  # noqa: E402


EXT_MACROS = OUT / "extended_auto_numbers.tex"


def ensure_dirs() -> None:
    for path in [CSV_DIR, FIG_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def objective(instance: PricingInstance, selections: Sequence[int]) -> float:
    return float(sum(instance.items[i][j].value for i, j in enumerate(selections)))


def selected_prices(instance: PricingInstance, selections: Sequence[int]) -> np.ndarray:
    vals = []
    for group, j in zip(instance.items, selections):
        price = group[int(j)].price
        vals.append(float(price) if price is not None else float(j))
    return np.array(vals, dtype=float)


def hull_vertex_to_options(hulls, vertices: Sequence[int]) -> List[int]:
    return [int(hull.option_indices[int(v)]) for hull, v in zip(hulls, vertices)]


def fixed_theta_recovery_variants(instance: PricingInstance, theta: float) -> Dict[str, object]:
    values, margins, uncertainties = option_arrays(instance)
    hulls, s_star_sum = compute_hulls_for_theta(values, margins, uncertainties, theta)
    capacity = float(s_star_sum - instance.gamma * theta)
    lp = greedy_lp(hulls, capacity)

    rd_obj = float(round_down_value(lp, hulls, values))
    rd_vertices = []
    for pos in lp.positions:
        rd_vertices.append(pos.upper_vertex if pos.lambda_ >= 1.0 - EPS else pos.lower_vertex)
    rd_sel = hull_vertex_to_options(hulls, rd_vertices)

    repaired = round_lp_solution(lp, hulls, capacity, upgrade_completion=False)
    if repaired is None:
        repaired_obj = float("nan")
        repaired_cert = float("nan")
    else:
        repaired_sel = hull_vertex_to_options(hulls, repaired.vertices)
        repaired_obj = float(repaired.value)
        repaired_cert = float(compute_certificate(instance, repaired_sel))

    return {
        "lp_objective": float(lp.lp_value),
        "round_down_objective": rd_obj,
        "round_down_certificate": float(compute_certificate(instance, rd_sel)),
        "repair_no_completion_objective": repaired_obj,
        "repair_no_completion_certificate": repaired_cert,
    }


def run_ablation(smoke: bool) -> List[Dict[str, object]]:
    seeds = [6101] if smoke else list(range(6101, 6107))
    n = 120 if smoke else 200
    m = 10 if smoke else 20
    gammas = [int(math.floor(math.sqrt(n)))] if smoke else [int(math.floor(math.sqrt(n))), int(math.floor(0.1 * n))]
    rows: List[Dict[str, object]] = []
    total = len(seeds) * len(gammas)
    done = 0
    for seed in seeds:
        master = make_master_portfolio(seed=seed, n_max=n, m_max=m, min_admissible_menu=min(10, m))
        for gamma in gammas:
            done += 1
            print(f"[extended-ablation {done}/{total}] seed={seed} n={n} m={m} gamma={gamma}", flush=True)
            prefix = build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma)
            full = solve(prefix.instance, upgrade_completion=True)
            no_completion = solve(prefix.instance, upgrade_completion=False)
            fixed = fixed_theta_recovery_variants(prefix.instance, float(full.theta))
            final_obj = float(full.objective)
            variants = [
                ("round_down_selected_theta", fixed["round_down_objective"], fixed["round_down_certificate"]),
                ("repair_no_completion_selected_theta", fixed["repair_no_completion_objective"], fixed["repair_no_completion_certificate"]),
                ("global_no_completion", float(no_completion.objective), float(no_completion.certificate_value)),
                ("HullRound", final_obj, float(full.certificate_value)),
            ]
            for variant, obj, cert in variants:
                rows.append(
                    {
                        "seed": seed,
                        "n": n,
                        "m": m,
                        "alpha": 0.10,
                        "gamma": gamma,
                        "gamma_regime": gamma_regime(n, gamma),
                        "variant": variant,
                        "objective": float(obj),
                        "relative_to_hullround": float(obj) / final_obj if final_obj > EPS else float("nan"),
                        "loss_vs_hullround": (final_obj - float(obj)) / final_obj if final_obj > EPS else float("nan"),
                        "certificate_value": float(cert),
                        "theta": float(full.theta),
                        "lp_objective": float(fixed["lp_objective"]),
                    }
                )
    write_csv(CSV_DIR / "extended_ablation.csv", rows)
    return rows


def run_time_quality(smoke: bool, solver_time_limit: float) -> List[Dict[str, object]]:
    seeds = [6201] if smoke else list(range(6201, 6204))
    n = 80 if smoke else 120
    m = 10
    gamma = int(math.floor(math.sqrt(n)))
    limits = [0.2, 1.0] if smoke else [0.2, 1.0, 5.0, min(float(solver_time_limit), 20.0)]
    rows: List[Dict[str, object]] = []
    total = len(seeds)
    for idx, seed in enumerate(seeds, 1):
        print(f"[extended-time-quality {idx}/{total}] seed={seed} n={n} gamma={gamma}", flush=True)
        master = make_master_portfolio(seed=seed, n_max=n, m_max=m, min_admissible_menu=min(10, m))
        prefix = build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma)
        hr = solve(prefix.instance, upgrade_completion=True)
        best_seen = float(hr.objective)
        scip_results: List[Tuple[float, Dict[str, object]]] = []
        for limit in limits:
            res = solve_full_robust_scip(prefix.instance, time_limit=float(limit), threads=1)
            scip_results.append((float(limit), res))
            if math.isfinite(float(res.get("objective", float("nan")))):
                best_seen = max(best_seen, float(res["objective"]))
        rows.append(
            {
                "seed": seed,
                "n": n,
                "m": m,
                "gamma": gamma,
                "method": "HullRound",
                "time_limit_s": 0.0,
                "runtime_s": float(hr.elapsed),
                "status": "CERTIFIED_HEURISTIC",
                "certified_optimal": False,
                "objective": float(hr.objective),
                "best_bound": float("nan"),
                "mip_gap": float("nan"),
                "relative_deficit_to_best_seen": (best_seen - float(hr.objective)) / best_seen if best_seen > EPS else 0.0,
                "certificate_value": float(hr.certificate_value),
            }
        )
        for limit, res in scip_results:
            obj = float(res.get("objective", float("nan")))
            rows.append(
                {
                    "seed": seed,
                    "n": n,
                    "m": m,
                    "gamma": gamma,
                    "method": "SCIP-FullRobust",
                    "time_limit_s": limit,
                    "runtime_s": float(res.get("runtime_s", float("nan"))),
                    "status": res.get("status"),
                    "certified_optimal": bool(res.get("certified")),
                    "objective": obj,
                    "best_bound": float(res.get("best_bound", float("nan"))),
                    "mip_gap": float(res.get("mip_gap", float("nan"))),
                    "relative_deficit_to_best_seen": (best_seen - obj) / best_seen if best_seen > EPS and math.isfinite(obj) else float("nan"),
                    "certificate_value": float(res.get("certificate_value", float("nan"))),
                }
            )
    write_csv(CSV_DIR / "extended_time_quality.csv", rows)
    return rows


def run_large_time_limited(smoke: bool, solver_time_limit: float) -> List[Dict[str, object]]:
    seeds = [6301] if smoke else [6301, 6302]
    ns = [200] if smoke else [200, 500]
    m = 10 if smoke else 20
    rows: List[Dict[str, object]] = []
    total = len(seeds) * len(ns)
    done = 0
    for seed in seeds:
        master = make_master_portfolio(seed=seed, n_max=max(ns), m_max=m, min_admissible_menu=min(10, m))
        for n in ns:
            done += 1
            gamma = int(math.floor(math.sqrt(n)))
            print(f"[extended-large-solver {done}/{total}] seed={seed} n={n} m={m} gamma={gamma}", flush=True)
            prefix = build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma)
            hr = hullround_metrics(prefix.instance, validate_lp=False)
            scip = solve_full_robust_scip(prefix.instance, time_limit=float(solver_time_limit), threads=1)
            scip_obj = float(scip.get("objective", float("nan")))
            rows.append(
                {
                    "seed": seed,
                    "n": n,
                    "m": m,
                    "alpha": 0.10,
                    "gamma": gamma,
                    "hullround_objective": float(hr["final_objective"]),
                    "hullround_runtime_s": float(hr["runtime_total_s"]),
                    "hullround_certificate": float(hr["certificate_value"]),
                    "candidate_count_reduced": int(hr["candidate_count_reduced"]),
                    "theta_evaluated_count": int(hr["theta_evaluated_count"]),
                    "total_hull_rebuilds": int(hr["total_hull_rebuilds"]),
                    "scip_time_limit_s": float(solver_time_limit),
                    "scip_status": scip.get("status"),
                    "scip_certified": bool(scip.get("certified")),
                    "scip_objective": scip_obj,
                    "scip_runtime_s": float(scip.get("runtime_s", float("nan"))),
                    "scip_best_bound": float(scip.get("best_bound", float("nan"))),
                    "scip_mip_gap": float(scip.get("mip_gap", float("nan"))),
                    "scip_certificate": float(scip.get("certificate_value", float("nan"))),
                    "scip_incumbent_minus_hullround_pct": (scip_obj - float(hr["final_objective"])) / float(hr["final_objective"])
                    if math.isfinite(scip_obj) and abs(float(hr["final_objective"])) > EPS
                    else float("nan"),
                }
            )
    write_csv(CSV_DIR / "extended_large_time_limited.csv", rows)
    return rows


def run_sensitivity_grid(smoke: bool) -> List[Dict[str, object]]:
    seeds = [6401] if smoke else [6401, 6402]
    n = 60 if smoke else 120
    menus = [10] if smoke else [10, 20, 50]
    alphas = [0.10] if smoke else [0.05, 0.10, 0.20, 0.30]
    gamma_grid = sorted(set([0, int(math.floor(math.sqrt(n))), int(math.floor(0.1 * n)), int(math.floor(0.25 * n)), n]))
    rows: List[Dict[str, object]] = []
    total = len(seeds) * len(menus) * len(alphas) * len(gamma_grid)
    done = 0
    for seed in seeds:
        master = make_master_portfolio(seed=seed, n_max=n, m_max=max(menus), min_admissible_menu=10)
        for m in menus:
            for alpha in alphas:
                base_obj = float("nan")
                for gamma in gamma_grid:
                    done += 1
                    print(
                        f"[extended-sensitivity {done}/{total}] seed={seed} n={n} m={m} alpha={alpha:.2f} gamma={gamma}",
                        flush=True,
                    )
                    prefix = build_prefix_instance(master, n=n, m=m, alpha=alpha, gamma=gamma)
                    hm = hullround_metrics(prefix.instance, validate_lp=False)
                    if gamma == 0:
                        base_obj = float(hm["final_objective"])
                    rows.append(
                        {
                            "seed": seed,
                            "n": n,
                            "m": m,
                            "alpha": alpha,
                            "gamma": gamma,
                            "gamma_frac": gamma / n,
                            "gamma_regime": gamma_regime(n, gamma),
                            "objective": float(hm["final_objective"]),
                            "revenue_ratio_vs_gamma0": float(hm["final_objective"]) / base_obj
                            if math.isfinite(base_obj) and base_obj > EPS
                            else float("nan"),
                            "certificate_value": float(hm["certificate_value"]),
                            "runtime_total_s": float(hm["runtime_total_s"]),
                            "gap_lp": float(hm["gap_lp"]),
                            "l_rd_over_delta": float(hm["l_rd_over_delta"]),
                            "candidate_count_reduced": int(hm["candidate_count_reduced"]),
                            "theta_evaluated_count": int(hm["theta_evaluated_count"]),
                            "total_hull_rebuilds": int(hm["total_hull_rebuilds"]),
                        }
                    )
    write_csv(CSV_DIR / "extended_sensitivity_grid.csv", rows)
    return rows


def stress_margins(prefix, selections: Sequence[int], protocol: str, rng: np.random.Generator, scenarios: int) -> Tuple[float, float]:
    data = selected_option_data(prefix, selections)
    g = data["g_hat"]
    d = data["delta"]
    k = data["k"]
    n = g.size
    scale = 1.0
    if protocol == "misspecified_alpha_2x":
        scale = 2.0
        protocol = "iid_uniform"
    if protocol == "iid_uniform":
        xi = rng.uniform(-1.0, 1.0, size=(scenarios, n))
    elif protocol == "correlated_common":
        common = rng.uniform(-1.0, 1.0, size=(scenarios, 1))
        local = rng.uniform(-1.0, 1.0, size=(scenarios, n))
        xi = np.clip(0.70 * common + 0.30 * local, -1.0, 1.0)
    elif protocol == "block_common":
        blocks = 4
        block_xi = rng.uniform(-1.0, 1.0, size=(scenarios, blocks))
        block_ids = np.minimum((np.arange(n) * blocks) // n, blocks - 1)
        local = rng.uniform(-1.0, 1.0, size=(scenarios, n))
        xi = np.clip(0.80 * block_xi[:, block_ids] + 0.20 * local, -1.0, 1.0)
    elif protocol == "heavy_tailed_clipped":
        xi = np.clip(rng.standard_t(df=3, size=(scenarios, n)) / 2.0, -1.0, 1.0)
    else:
        raise ValueError(f"unknown stress protocol: {protocol}")
    realized = np.maximum(0.0, g[None, :] + xi * (scale * d[None, :]))
    margins = realized @ k
    return float(np.mean(margins < -EPS)), float(np.quantile(margins, 0.05))


def run_out_of_model_stress(smoke: bool, scenarios: int) -> List[Dict[str, object]]:
    seeds = [6501] if smoke else list(range(6501, 6505))
    n = 80 if smoke else 120
    m = 10
    protocols = ["iid_uniform"] if smoke else ["iid_uniform", "correlated_common", "block_common", "heavy_tailed_clipped", "misspecified_alpha_2x"]
    gamma_values = [0, int(math.floor(math.sqrt(n)))]
    rows: List[Dict[str, object]] = []
    total = len(seeds) * len(gamma_values)
    done = 0
    for seed in seeds:
        master = make_master_portfolio(seed=seed, n_max=n, m_max=m, min_admissible_menu=min(10, m))
        for gamma in gamma_values:
            done += 1
            print(f"[extended-stress {done}/{total}] seed={seed} n={n} gamma={gamma}", flush=True)
            prefix = build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma)
            sol = solve(prefix.instance, upgrade_completion=True)
            for protocol in protocols:
                rng = np.random.default_rng(202406 + seed + 31 * gamma + 17 * len(protocol))
                viol, q05 = stress_margins(prefix, sol.selections, protocol, rng, scenarios)
                rows.append(
                    {
                        "seed": seed,
                        "n": n,
                        "m": m,
                        "alpha_solve": 0.10,
                        "gamma": gamma,
                        "protocol": protocol,
                        "scenarios": scenarios,
                        "objective": float(sol.objective),
                        "certificate_value": float(sol.certificate_value),
                        "violation_probability": viol,
                        "q05_margin": q05,
                    }
                )
    write_csv(CSV_DIR / "extended_out_of_model_stress.csv", rows)
    return rows


def make_retail_decision_path() -> List[Dict[str, object]]:
    retail = read_csv(CSV_DIR / "retail_replicates.csv")
    segment = read_csv(CSV_DIR / "retail_segment_adjustments.csv")
    agg_abs_by_seed_gamma: Dict[Tuple[int, int], float] = {}
    for row in segment:
        key = (int(row["seed"]), int(row["gamma"]))
        # Weighted segment sizes are fixed by the retail generator.
        weights = {"Staples": 120, "Mainstream": 100, "Premium": 50, "Private label": 30}
        agg_abs_by_seed_gamma[key] = agg_abs_by_seed_gamma.get(key, 0.0) + float(row["avg_abs_price_change"]) * weights[row["segment"]] / 300.0
    rows: List[Dict[str, object]] = []
    for row in retail:
        key = (int(row["seed"]), int(row["gamma"]))
        rows.append(
            {
                "seed": int(row["seed"]),
                "gamma": int(row["gamma"]),
                "gamma_frac": float(row["gamma_frac"]),
                "revenue_ratio": float(row["revenue_ratio"]),
                "price_changes_vs_gamma0": float(row["price_changes_vs_gamma0"]),
                "share_changed": float(row["price_changes_vs_gamma0"]) / 300.0,
                "avg_abs_price_change": agg_abs_by_seed_gamma.get(key, float("nan")),
                "z_gamma": float(row["z_gamma"]),
                "violation_iid": float(row["violation_iid"]),
            }
        )
    write_csv(CSV_DIR / "extended_retail_decision_path.csv", rows)
    return rows


def make_bound_tightness() -> List[Dict[str, object]]:
    gap_rows = read_csv(CSV_DIR / "synthetic_gap_replicates.csv")
    rows = sorted(gap_rows, key=lambda r: float(r["l_rd_over_delta"]), reverse=True)[:8]
    out: List[Dict[str, object]] = []
    for rank, r in enumerate(rows, 1):
        out.append(
            {
                "rank": rank,
                "seed": int(r["seed"]),
                "n": int(r["n"]),
                "m": int(r["m"]),
                "gamma": int(r["gamma"]),
                "gamma_regime": r["gamma_regime"],
                "theta": float(r["theta"]),
                "l_rd": float(r["l_rd"]),
                "delta_v_max_theta": float(r["delta_v_max_theta"]),
                "l_rd_over_delta": float(r["l_rd_over_delta"]),
                "gap_lp": float(r["gap_lp"]),
            }
        )
    write_csv(CSV_DIR / "extended_bound_tightness.csv", out)
    return out


def run_gap_scaling(smoke: bool) -> List[Dict[str, object]]:
    seeds = [6601] if smoke else list(range(6601, 6604))
    ns = [50, 100] if smoke else [50, 100, 200, 500]
    m = 8
    rows: List[Dict[str, object]] = []
    total = len(seeds) * len(ns)
    done = 0
    for seed in seeds:
        master = make_master_portfolio(seed=seed, n_max=max(ns), m_max=m, min_admissible_menu=min(8, m))
        for n in ns:
            done += 1
            gamma = int(math.floor(math.sqrt(n)))
            print(f"[extended-gap-scaling {done}/{total}] seed={seed} n={n} m={m} gamma={gamma}", flush=True)
            prefix = build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=gamma)
            hm = hullround_metrics(prefix.instance, validate_lp=False)
            hm.pop("selections", None)
            rows.append(
                {
                    "seed": seed,
                    "n": n,
                    "m": m,
                    "alpha": 0.10,
                    "gamma": gamma,
                    "gamma_regime": gamma_regime(n, gamma),
                    "gap_lp": float(hm["gap_lp"]),
                    "n_gap_lp": n * float(hm["gap_lp"]),
                    "l_rd_over_delta": float(hm["l_rd_over_delta"]),
                    "runtime_total_s": float(hm["runtime_total_s"]),
                    "candidate_count_reduced": int(hm["candidate_count_reduced"]),
                    "theta_evaluated_count": int(hm["theta_evaluated_count"]),
                }
            )
    write_csv(CSV_DIR / "extended_gap_scaling.csv", rows)
    return rows


def run_method_frontier(smoke: bool, scenarios: int, solver_time_limit: float) -> List[Dict[str, object]]:
    seeds = [6701] if smoke else list(range(6701, 6704))
    n = 60 if smoke else 120
    m = 8 if smoke else 10
    alpha = 0.10
    gamma_grid = [0, int(math.floor(math.sqrt(n)))] if smoke else sorted(
        set([0, 1, 3, 5, 10, int(math.floor(math.sqrt(n))), int(math.floor(0.1 * n)), int(math.floor(0.25 * n)), int(math.floor(0.5 * n)), n])
    )
    rows: List[Dict[str, object]] = []
    total = len(seeds) * len(gamma_grid)
    done = 0
    for seed in seeds:
        master = make_master_portfolio(seed=seed, n_max=n, m_max=m, min_admissible_menu=min(10, m))
        prefix_nominal = build_prefix_instance(master, n=n, m=m, alpha=alpha, gamma=0)
        nominal_metrics = hullround_metrics(prefix_nominal.instance, validate_lp=False)
        nominal_selections = list(nominal_metrics.pop("selections"))
        nominal_objective = objective(prefix_nominal.instance, nominal_selections)

        prefix_box = build_prefix_instance(master, n=n, m=m, alpha=alpha, gamma=n)
        box_metrics = hullround_metrics(prefix_box.instance, validate_lp=False)
        box_selections = list(box_metrics.pop("selections"))

        for gamma in gamma_grid:
            done += 1
            print(f"[method-frontier {done}/{total}] seed={seed} gamma={gamma}", flush=True)
            prefix = build_prefix_instance(master, n=n, m=m, alpha=alpha, gamma=gamma)
            method_selections: List[Tuple[str, Sequence[int], str, bool, float]] = [
                ("Nominal", nominal_selections, "CERTIFIED_HEURISTIC", True, float(nominal_metrics["runtime_total_s"])),
                ("Box robust", box_selections, "CERTIFIED_HEURISTIC", True, float(box_metrics["runtime_total_s"])),
            ]

            hr = hullround_metrics(prefix.instance, validate_lp=False)
            hr_selections = list(hr.pop("selections"))
            method_selections.append(("HullRound", hr_selections, "CERTIFIED_HEURISTIC", True, float(hr["runtime_total_s"])))

            scip = solve_full_robust_scip(prefix.instance, time_limit=float(solver_time_limit), threads=1)
            if scip.get("selections") is not None:
                method_selections.append(
                    (
                        "SCIP-FullRobust",
                        list(scip["selections"]),  # type: ignore[arg-type]
                        str(scip.get("status")),
                        bool(scip.get("certified")),
                        float(scip.get("runtime_s", float("nan"))),
                    )
                )

            for method, selections, status, certified, runtime_s in method_selections:
                rng = np.random.default_rng(np.random.SeedSequence([seed, gamma, len(method), 6701]))
                violation, q05_margin = simulate_iid_prefix(prefix, selections, rng, scenarios=scenarios if not smoke else min(400, scenarios))
                z_gamma = robust_residual_for_budget(prefix.instance, selections, gamma)
                obj = objective(prefix.instance, selections)
                rows.append(
                    {
                        "seed": seed,
                        "n": n,
                        "m": m,
                        "alpha": alpha,
                        "gamma": gamma,
                        "gamma_frac": gamma / n,
                        "method": method,
                        "status": status,
                        "certified": certified,
                        "objective": obj,
                        "revenue_ratio_vs_nominal": obj / nominal_objective if nominal_objective > EPS else float("nan"),
                        "z_gamma": z_gamma,
                        "violation_iid": violation,
                        "q05_margin_iid": q05_margin,
                        "runtime_s": runtime_s,
                    }
                )
    write_csv(CSV_DIR / "extended_method_frontier.csv", rows)
    return rows


def calibration_iid(prefix, selections: Sequence[int], true_alpha: float, rng: np.random.Generator, scenarios: int) -> Tuple[float, float]:
    data = selected_prefix_arrays(prefix, selections)
    g = data["g_hat"]
    k = data["k"]
    true_delta = true_alpha * g
    margins: List[np.ndarray] = []
    batch = 1000
    for start in range(0, scenarios, batch):
        b = min(batch, scenarios - start)
        xi = rng.uniform(-1.0, 1.0, size=(b, len(selections)))
        realized = np.maximum(0.0, g[None, :] + xi * true_delta[None, :])
        margins.append(realized @ k)
    arr = np.concatenate(margins)
    return float(np.mean(arr < -EPS)), float(np.quantile(arr, 0.05))


def run_calibration_grid(smoke: bool, scenarios: int) -> List[Dict[str, object]]:
    seeds = [6801] if smoke else list(range(6801, 6804))
    n = 60 if smoke else 120
    m = 10
    gamma = int(math.floor(math.sqrt(n)))
    model_alphas = [0.10] if smoke else [0.05, 0.10, 0.20, 0.30]
    true_alphas = [0.10] if smoke else [0.05, 0.10, 0.20, 0.30, 0.40]
    rows: List[Dict[str, object]] = []
    total = len(seeds) * len(model_alphas)
    done = 0
    for seed in seeds:
        master = make_master_portfolio(seed=seed, n_max=n, m_max=m, min_admissible_menu=min(10, m))
        nominal_prefix = build_prefix_instance(master, n=n, m=m, alpha=0.10, gamma=0)
        nominal = hullround_metrics(nominal_prefix.instance, validate_lp=False)
        nominal_obj = float(nominal["final_objective"])
        for model_alpha in model_alphas:
            done += 1
            print(f"[calibration-grid {done}/{total}] seed={seed} model_alpha={model_alpha:.2f}", flush=True)
            prefix = build_prefix_instance(master, n=n, m=m, alpha=model_alpha, gamma=gamma)
            hm = hullround_metrics(prefix.instance, validate_lp=False)
            selections = list(hm.pop("selections"))
            for true_alpha in true_alphas:
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(model_alpha * 100), int(true_alpha * 100), 6801]))
                violation, q05_margin = calibration_iid(prefix, selections, true_alpha, rng, scenarios if not smoke else min(400, scenarios))
                rows.append(
                    {
                        "seed": seed,
                        "n": n,
                        "m": m,
                        "gamma": gamma,
                        "model_alpha": model_alpha,
                        "true_alpha": true_alpha,
                        "objective": float(hm["final_objective"]),
                        "revenue_ratio_vs_nominal": float(hm["final_objective"]) / nominal_obj if nominal_obj > EPS else float("nan"),
                        "certificate_model": float(hm["certificate_value"]),
                        "violation_iid": violation,
                        "q05_margin_iid": q05_margin,
                    }
                )
    write_csv(CSV_DIR / "extended_calibration_grid.csv", rows)
    return rows


def plot_ablation(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    variants = [
        ("round_down_selected_theta", "Round-down"),
        ("repair_no_completion_selected_theta", "Repair"),
        ("global_no_completion", "No completion"),
        ("HullRound", "HullRound"),
    ]
    fig, ax = plt.subplots(figsize=(4.7, 2.35))
    data = [[100.0 * float(r["loss_vs_hullround"]) for r in rows if r["variant"] == key] for key, _ in variants]
    ax.boxplot(data, tick_labels=[label for _, label in variants], showfliers=False)
    ax.axhline(0.0, color="#222222", linestyle=":", linewidth=0.9)
    ax.set_ylabel("Loss vs. HullRound (%)")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "extended_ablation_quality.pdf")
    plt.close(fig)


def plot_extended_gap_validation(gap_rows: List[Dict[str, object]], scale_rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.35))

    regimes = list(dict.fromkeys(str(r["gamma_regime"]) for r in gap_rows))
    for regime in regimes:
        group = [r for r in gap_rows if str(r["gamma_regime"]) == regime]
        ns = sorted({int(r["n"]) for r in group})
        y = [median([float(r["l_rd_over_delta"]) for r in group if int(r["n"]) == n]) for n in ns]
        lo = [q([float(r["l_rd_over_delta"]) for r in group if int(r["n"]) == n], 0.25) for n in ns]
        hi = [q([float(r["l_rd_over_delta"]) for r in group if int(r["n"]) == n], 0.75) for n in ns]
        axes[0].plot(ns, y, marker="o", label=regime)
        axes[0].fill_between(ns, lo, hi, alpha=0.14, linewidth=0)
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("$n$")
    axes[0].set_ylabel(r"$L_{\mathrm{rd}}/\Delta V_{\max}^{\theta}$")
    axes[0].set_title("(a) Bound tightness")
    axes[0].legend(loc="best")

    ns = sorted({int(r["n"]) for r in scale_rows})
    med_gap = [median([float(r["gap_lp"]) for r in scale_rows if int(r["n"]) == n]) for n in ns]
    lo_gap = [q([float(r["gap_lp"]) for r in scale_rows if int(r["n"]) == n], 0.25) for n in ns]
    hi_gap = [q([float(r["gap_lp"]) for r in scale_rows if int(r["n"]) == n], 0.75) for n in ns]
    axes[1].plot(ns, med_gap, marker="o", color="#4c78a8", label="Observed")
    axes[1].fill_between(ns, lo_gap, hi_gap, color="#4c78a8", alpha=0.16, linewidth=0)
    if med_gap and med_gap[0] > 0:
        ref = [med_gap[0] * ns[0] / n for n in ns]
        axes[1].plot(ns, ref, color="#e45756", linestyle="--", label="$1/n$ reference")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("$n$")
    axes[1].set_ylabel(r"$\mathrm{Gap}_{\mathrm{LP}}$")
    axes[1].set_title("(b) Relative gap")
    axes[1].legend(loc="best")

    med_scaled = [median([float(r["n_gap_lp"]) for r in scale_rows if int(r["n"]) == n]) for n in ns]
    lo_scaled = [q([float(r["n_gap_lp"]) for r in scale_rows if int(r["n"]) == n], 0.25) for n in ns]
    hi_scaled = [q([float(r["n_gap_lp"]) for r in scale_rows if int(r["n"]) == n], 0.75) for n in ns]
    axes[2].plot(ns, med_scaled, marker="o", color="#54a24b")
    axes[2].fill_between(ns, lo_scaled, hi_scaled, color="#54a24b", alpha=0.16, linewidth=0)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("$n$")
    axes[2].set_ylabel(r"$n\times\mathrm{Gap}_{\mathrm{LP}}$")
    axes[2].set_title("(c) Scaled gap")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gap_validation_extended.pdf")
    plt.close(fig)


def plot_method_frontier(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    methods = ["Nominal", "HullRound", "SCIP-FullRobust", "Box robust"]
    colors = {
        "Nominal": "#bab0ac",
        "HullRound": "#4c78a8",
        "SCIP-FullRobust": "#e45756",
        "Box robust": "#54a24b",
    }
    styles = {"Nominal": ":", "HullRound": "-", "SCIP-FullRobust": "--", "Box robust": "-."}
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.4), sharex=True)
    panels = [
        (axes[0], "revenue_ratio_vs_nominal", "Revenue ratio", "(a) Revenue"),
        (axes[1], "z_gamma", r"$Z_\Gamma(\mathbf{x})$", "(b) Certificate"),
        (axes[2], "violation_iid", "IID violation probability", "(c) IID stress"),
    ]
    for method in methods:
        group = [r for r in rows if str(r["method"]) == method]
        if not group:
            continue
        n_ref = max(int(r["n"]) for r in group)
        gammas = sorted({int(r["gamma"]) for r in group})
        x = np.array([g / n_ref for g in gammas])
        for ax, col, ylabel, title in panels:
            med = [median([float(r[col]) for r in group if int(r["gamma"]) == g]) for g in gammas]
            lo = [q([float(r[col]) for r in group if int(r["gamma"]) == g], 0.25) for g in gammas]
            hi = [q([float(r[col]) for r in group if int(r["gamma"]) == g], 0.75) for g in gammas]
            ax.plot(x, med, marker="o", color=colors[method], linestyle=styles[method], label=method)
            ax.fill_between(x, lo, hi, color=colors[method], alpha=0.10, linewidth=0)
            ax.set_xlabel(r"$\Gamma/n$")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
    for ax in axes:
        ax.axvline(math.sqrt(max(int(r["n"]) for r in rows)) / max(int(r["n"]) for r in rows), color="0.45", linestyle=":", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_ylim(bottom=0.0)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "method_robustness_frontier.pdf")
    plt.close(fig)


def plot_time_quality(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(4.7, 2.35))
    scip = [r for r in rows if r["method"] == "SCIP-FullRobust"]
    limits = sorted({float(r["time_limit_s"]) for r in scip})
    med_def = [median([float(r["relative_deficit_to_best_seen"]) for r in scip if float(r["time_limit_s"]) == lim]) for lim in limits]
    cert_share = [np.mean([r["certified_optimal"] == "True" or r["certified_optimal"] is True for r in scip if float(r["time_limit_s"]) == lim]) for lim in limits]
    hr_def = median([float(r["relative_deficit_to_best_seen"]) for r in rows if r["method"] == "HullRound"])
    ax.plot(limits, med_def, marker="o", label="SCIP incumbent deficit")
    ax.axhline(hr_def, color="#54a24b", linestyle="--", label="HullRound deficit")
    ax.set_xscale("log")
    ax.set_ylabel("Deficit to best observed")
    ax.set_xlabel("SCIP time limit (s)")
    ax2 = ax.twinx()
    ax2.plot(limits, cert_share, marker="s", color="#e45756", linestyle=":", label="SCIP certified share")
    ax2.set_ylabel("Certified share")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "extended_time_quality.pdf")
    plt.close(fig)


def plot_large_time_limited(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ns = sorted({int(r["n"]) for r in rows})
    hr = [median([float(r["hullround_runtime_s"]) for r in rows if int(r["n"]) == n]) for n in ns]
    scip = [median([float(r["scip_runtime_s"]) for r in rows if int(r["n"]) == n]) for n in ns]
    axes[0].plot(ns, hr, marker="o", label="HullRound")
    axes[0].plot(ns, scip, marker="s", linestyle="--", label="SCIP limited")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("$n$")
    axes[0].set_ylabel("Seconds")
    axes[0].legend(loc="best")
    inc = [median([float(r["scip_incumbent_minus_hullround_pct"]) for r in rows if int(r["n"]) == n]) for n in ns]
    axes[1].axhline(0.0, color="#222222", linestyle=":", linewidth=1.0)
    axes[1].plot(ns, [100.0 * x for x in inc], marker="o", color="#e45756")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("$n$")
    axes[1].set_ylabel("SCIP - HR (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "extended_large_time_limited.pdf")
    plt.close(fig)


def plot_sensitivity(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    alphas = sorted({float(r["alpha"]) for r in rows})
    menus = sorted({int(r["m"]) for r in rows})
    n = int(rows[0]["n"])
    sqrt_gamma = int(math.floor(math.sqrt(n)))
    full_gamma = n
    fig, axes = plt.subplots(1, 2, figsize=(6.45, 2.55), sharey=True)
    for ax, gamma, title in [(axes[0], sqrt_gamma, r"$\Gamma=\lfloor\sqrt{n}\rfloor$"), (axes[1], full_gamma, r"$\Gamma=n$")]:
        mat = np.zeros((len(alphas), len(menus)))
        for i, alpha in enumerate(alphas):
            for j, m in enumerate(menus):
                vals = [
                    float(r["revenue_ratio_vs_gamma0"])
                    for r in rows
                    if float(r["alpha"]) == alpha and int(r["m"]) == m and int(r["gamma"]) == gamma
                ]
                mat[i, j] = median(vals)
        im = ax.imshow(mat, vmin=max(0.97, float(np.nanmin(mat)) - 0.002), vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(menus)), [str(m) for m in menus])
        ax.set_yticks(range(len(alphas)), [f"{a:.2f}" for a in alphas])
        ax.set_xlabel("Menu size $m$")
        for i in range(len(alphas)):
            for j in range(len(menus)):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=7)
    axes[0].set_ylabel(r"Uncertainty scale $\alpha$")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="Revenue ratio")
    fig.savefig(FIG_DIR / "extended_sensitivity_grid.pdf")
    plt.close(fig)


def plot_theta_decomposition(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    def runtime(row: Dict[str, object]) -> float:
        if "runtime_total_s" in row:
            return float(row["runtime_total_s"])
        return float(row["hullround_runtime_s"])

    fig, axes = plt.subplots(1, 2, figsize=(6.55, 2.4))
    for n in sorted({int(r["n"]) for r in rows}):
        group = [r for r in rows if int(r["n"]) == n]
        axes[0].scatter(
            [float(r["theta_evaluated_count"]) for r in group],
            [runtime(r) for r in group],
            s=14,
            label=f"$n={n}$",
            alpha=0.75,
        )
    axes[0].set_xlabel("Evaluated $\\theta$ values")
    axes[0].set_ylabel("Seconds")
    axes[0].set_yscale("log")
    axes[0].legend(ncols=2, loc="best")
    ns = sorted({int(r["n"]) for r in rows})
    evals = [median([float(r["theta_evaluated_count"]) for r in rows if int(r["n"]) == n]) for n in ns]
    rebuilds = [median([float(r["total_hull_rebuilds"]) for r in rows if int(r["n"]) == n]) for n in ns]
    axes[1].plot(ns, evals, marker="o", label="Theta values")
    axes[1].plot(ns, rebuilds, marker="s", linestyle="--", label="Hull rebuilds")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("$n$")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "extended_theta_decomposition.pdf")
    plt.close(fig)


def plot_computational_scaling(scale_rows: List[Dict[str, object]], theta_rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()

    def row_runtime(row: Dict[str, object]) -> float:
        if "runtime_total_s" in row:
            return float(row["runtime_total_s"])
        return float(row["hullround_runtime_s"])

    ns = sorted({int(r["n"]) for r in scale_rows})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.35))

    for m in sorted({int(r["m"]) for r in scale_rows}):
        runtime = [median([float(r["hullround_runtime_s"]) for r in scale_rows if int(r["n"]) == n and int(r["m"]) == m]) for n in ns]
        compression = [median([float(r["hull_compression_ratio"]) for r in scale_rows if int(r["n"]) == n and int(r["m"]) == m]) for n in ns]
        axes[0, 0].plot(ns, runtime, marker="o", label=fr"$m={m}$")
        axes[0, 1].plot(ns, compression, marker="o", label=fr"$m={m}$")

    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("$n$")
    axes[0, 0].set_ylabel("Seconds")
    axes[0, 0].set_title("(a) Runtime")
    axes[0, 0].legend(loc="best")

    axes[0, 1].set_xlabel("$n$")
    axes[0, 1].set_ylabel("Hull/raw options")
    axes[0, 1].set_title("(b) Hull compression")

    for n in sorted({int(r["n"]) for r in theta_rows}):
        group = [r for r in theta_rows if int(r["n"]) == n]
        axes[1, 0].scatter(
            [float(r["theta_evaluated_count"]) for r in group],
            [row_runtime(r) for r in group],
            s=14,
            label=f"$n={n}$",
            alpha=0.75,
        )
    axes[1, 0].set_xlabel("Evaluated $\\theta$ values")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("(c) Theta-loop cost")
    axes[1, 0].legend(ncols=2, loc="best")

    theta_ns = sorted({int(r["n"]) for r in theta_rows})
    evals = [median([float(r["theta_evaluated_count"]) for r in theta_rows if int(r["n"]) == n]) for n in theta_ns]
    rebuilds = [median([float(r["total_hull_rebuilds"]) for r in theta_rows if int(r["n"]) == n]) for n in theta_ns]
    axes[1, 1].plot(theta_ns, evals, marker="o", label="Theta values")
    axes[1, 1].plot(theta_ns, rebuilds, marker="s", linestyle="--", label="Hull rebuilds")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("$n$")
    axes[1, 1].set_title("(d) Loop size")
    axes[1, 1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "computational_scaling_diagnostics.pdf")
    plt.close(fig)


def plot_stress(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    protocols = list(dict.fromkeys(str(r["protocol"]) for r in rows))
    protocol_labels = {
        "iid_uniform": "IID",
        "correlated_common": "Common",
        "block_common": "Block",
        "heavy_tailed_clipped": "Heavy-tail",
        "misspecified_alpha_2x": r"$2\times\alpha$",
    }
    gammas = sorted({int(r["gamma"]) for r in rows})
    width = 0.36
    x = np.arange(len(protocols))
    fig, ax = plt.subplots(figsize=(5.8, 2.35))
    for offset, gamma in zip([-width / 2, width / 2], gammas):
        vals = [median([float(r["violation_probability"]) for r in rows if str(r["protocol"]) == p and int(r["gamma"]) == gamma]) for p in protocols]
        ax.bar(x + offset, [100.0 * v for v in vals], width=width, label=fr"$\Gamma={gamma}$")
    ax.set_xticks(x, [protocol_labels.get(p, p.replace("_", " ")) for p in protocols], rotation=0)
    ax.set_ylabel("Violation probability (%)")
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "extended_out_of_model_stress.pdf")
    plt.close(fig)


def plot_robustness_design_diagnostics(
    sensitivity: List[Dict[str, object]],
    stress: List[Dict[str, object]],
    calibration: List[Dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.45))

    alphas = sorted({float(r["alpha"]) for r in sensitivity})
    menus = sorted({int(r["m"]) for r in sensitivity})
    n = int(sensitivity[0]["n"])
    sqrt_gamma = int(math.floor(math.sqrt(n)))
    mat = np.zeros((len(alphas), len(menus)))
    for i, alpha in enumerate(alphas):
        for j, m in enumerate(menus):
            vals = [
                float(r["revenue_ratio_vs_gamma0"])
                for r in sensitivity
                if float(r["alpha"]) == alpha and int(r["m"]) == m and int(r["gamma"]) == sqrt_gamma
            ]
            mat[i, j] = median(vals)
    im0 = axes[0].imshow(mat, vmin=max(0.97, float(np.nanmin(mat)) - 0.002), vmax=1.0, cmap="viridis", aspect="auto")
    axes[0].set_title(r"(a) Revenue at $\lfloor\sqrt{n}\rfloor$")
    axes[0].set_xticks(range(len(menus)), [str(m) for m in menus])
    axes[0].set_yticks(range(len(alphas)), [f"{a:.2f}" for a in alphas])
    axes[0].set_xlabel("Menu size $m$")
    axes[0].set_ylabel(r"Modeled $\alpha$")
    for i in range(len(alphas)):
        for j in range(len(menus)):
            axes[0].text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white", fontsize=6.6)

    protocols = list(dict.fromkeys(str(r["protocol"]) for r in stress))
    protocol_labels = {
        "iid_uniform": "IID",
        "correlated_common": "Common",
        "block_common": "Block",
        "heavy_tailed_clipped": "Heavy-tail",
        "misspecified_alpha_2x": r"$2\times\alpha$",
    }
    gammas = sorted({int(r["gamma"]) for r in stress})
    width = 0.36
    x = np.arange(len(protocols))
    for offset, gamma in zip([-width / 2, width / 2], gammas):
        vals = [median([float(r["violation_probability"]) for r in stress if str(r["protocol"]) == p and int(r["gamma"]) == gamma]) for p in protocols]
        axes[1].bar(x + offset, [100.0 * v for v in vals], width=width, label=fr"$\Gamma={gamma}$")
    axes[1].set_xticks(x, [protocol_labels.get(p, p.replace("_", " ")) for p in protocols], rotation=18, ha="right")
    axes[1].set_ylabel("Violation probability (%)")
    axes[1].set_title("(b) Out-of-model stress")
    axes[1].legend(loc="upper right", ncols=2)

    model_alphas = sorted({float(r["model_alpha"]) for r in calibration})
    true_alphas = sorted({float(r["true_alpha"]) for r in calibration})
    cal = np.zeros((len(true_alphas), len(model_alphas)))
    for i, true_alpha in enumerate(true_alphas):
        for j, model_alpha in enumerate(model_alphas):
            vals = [
                float(r["violation_iid"])
                for r in calibration
                if abs(float(r["true_alpha"]) - true_alpha) < 1e-9 and abs(float(r["model_alpha"]) - model_alpha) < 1e-9
            ]
            cal[i, j] = 100.0 * median(vals)
    im2 = axes[2].imshow(cal, vmin=0.0, vmax=max(5.0, float(np.nanmax(cal))), cmap="magma_r", aspect="auto")
    axes[2].set_title("(c) Calibration heatmap")
    axes[2].set_xticks(range(len(model_alphas)), [f"{a:.2f}" for a in model_alphas])
    axes[2].set_yticks(range(len(true_alphas)), [f"{a:.2f}" for a in true_alphas])
    axes[2].set_xlabel(r"Modeled $\alpha$")
    axes[2].set_ylabel(r"True $\alpha$")
    for i in range(len(true_alphas)):
        for j in range(len(model_alphas)):
            axes[2].text(j, i, f"{cal[i, j]:.1f}", ha="center", va="center", color="black", fontsize=6.4)

    fig.colorbar(im0, ax=axes[0], shrink=0.78, label="Revenue ratio")
    fig.colorbar(im2, ax=axes[2], shrink=0.78, label="Violation (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "robustness_design_diagnostics.pdf")
    plt.close(fig)


def plot_retail_path(rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style()
    gammas = sorted({int(r["gamma"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), sharex=True)
    fig.subplots_adjust(wspace=0.36)
    for ax, col, ylabel in [
        (axes[0], "share_changed", "Share of prices changed"),
        (axes[1], "avg_abs_price_change", "Avg. |price change|"),
    ]:
        med = [median([float(r[col]) for r in rows if int(r["gamma"]) == g]) for g in gammas]
        lo = [q([float(r[col]) for r in rows if int(r["gamma"]) == g], 0.25) for g in gammas]
        hi = [q([float(r[col]) for r in rows if int(r["gamma"]) == g], 0.75) for g in gammas]
        ax.plot(gammas, med, marker="o", color="#4c78a8")
        ax.fill_between(gammas, lo, hi, color="#4c78a8", alpha=0.18, linewidth=0)
        ax.axvline(int(math.floor(math.sqrt(300))), color="#222222", linestyle=":", linewidth=1.0)
        ax.set_xlabel(r"$\Gamma$")
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "extended_retail_decision_path.pdf")
    plt.close(fig)


def plot_retail_decision_diagnostics(
    path_rows: List[Dict[str, object]],
    seg_rows: List[Dict[str, object]],
    conc_rows: List[Dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    apply_plot_style()
    gammas = sorted({int(r["gamma"]) for r in path_rows})
    gamma_star = int(math.floor(math.sqrt(300)))
    segs = ["Staples", "Mainstream", "Premium", "Private label"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    fig = plt.figure(figsize=(7.2, 4.35))
    gs = GridSpec(2, 6, figure=fig, hspace=0.62, wspace=0.82)
    ax_share = fig.add_subplot(gs[0, 0:3])
    ax_abs = fig.add_subplot(gs[0, 3:6])
    ax_signed = fig.add_subplot(gs[1, 0:2])
    ax_incidence = fig.add_subplot(gs[1, 2:4])
    ax_conc = fig.add_subplot(gs[1, 4:6])

    for ax, col, ylabel, title in [
        (ax_share, "share_changed", "Share changed", "(a) Decision breadth"),
        (ax_abs, "avg_abs_price_change", "Avg. |price change|", "(b) Decision magnitude"),
    ]:
        med = [median([float(r[col]) for r in path_rows if int(r["gamma"]) == g]) for g in gammas]
        lo = [q([float(r[col]) for r in path_rows if int(r["gamma"]) == g], 0.25) for g in gammas]
        hi = [q([float(r[col]) for r in path_rows if int(r["gamma"]) == g], 0.75) for g in gammas]
        ax.plot(gammas, med, marker="o", color="#4c78a8")
        ax.fill_between(gammas, lo, hi, color="#4c78a8", alpha=0.18, linewidth=0)
        ax.axvline(gamma_star, color="#222222", linestyle=":", linewidth=1.0)
        ax.set_xlabel(r"$\Gamma$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    xidx = np.arange(len(segs))
    signed = [[float(r["avg_signed_price_change"]) for r in seg_rows if int(r["gamma"]) == gamma_star and r["segment"] == s] for s in segs]
    changed = [[float(r["share_changed"]) for r in seg_rows if int(r["gamma"]) == gamma_star and r["segment"] == s] for s in segs]
    signed_med = [median(v) for v in signed]
    signed_err = np.array([[median(v) - q(v, 0.25) for v in signed], [q(v, 0.75) - median(v) for v in signed]])
    ax_signed.bar(xidx, signed_med, yerr=signed_err, color=colors, capsize=2, linewidth=0.6, edgecolor="0.25")
    ax_signed.axhline(0.0, color="black", linewidth=0.8)
    ax_signed.set_title("(c) Segment direction")
    ax_signed.set_ylabel("Signed change")

    changed_med = [median(v) for v in changed]
    changed_err = np.array([[median(v) - q(v, 0.25) for v in changed], [q(v, 0.75) - median(v) for v in changed]])
    ax_incidence.bar(xidx, changed_med, yerr=changed_err, color=colors, capsize=2, linewidth=0.6, edgecolor="0.25")
    ax_incidence.set_ylim(0.0, 1.0)
    ax_incidence.set_title("(d) Segment incidence")
    ax_incidence.set_ylabel("Share changed")

    for ax in [ax_signed, ax_incidence]:
        ax.set_xticks(xidx, segs, rotation=24, ha="right")

    x = np.array([g / 300 for g in gammas])
    for metric, label, color in [
        ("top10_margin_share", "top-10", "#4c78a8"),
        ("hhi_positive_margin", "HHI", "#f58518"),
        ("gini_positive_margin", "Gini", "#54a24b"),
    ]:
        y, lo, hi = [], [], []
        for gamma in gammas:
            vals = [float(r[metric]) for r in conc_rows if int(r["gamma"]) == gamma]
            y.append(median(vals))
            lo.append(q(vals, 0.25))
            hi.append(q(vals, 0.75))
        ax_conc.plot(x, y, marker="o", label=label, color=color)
        ax_conc.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
    ax_conc.axvline(gamma_star / 300, color="#222222", linestyle=":", linewidth=1.0)
    ax_conc.set_xlabel(r"$\Gamma/n$")
    ax_conc.set_ylabel("Concentration")
    ax_conc.set_title("(e) Margin concentration")
    ax_conc.legend(loc="best")

    fig.savefig(FIG_DIR / "retail_decision_diagnostics.pdf")
    plt.close(fig)


def write_tables(ablation: List[Dict[str, object]], large: List[Dict[str, object]], tight: List[Dict[str, object]]) -> None:
    variants = [
        ("round_down_selected_theta", "Round-down"),
        ("repair_no_completion_selected_theta", "Repair"),
        ("global_no_completion", "No completion"),
        ("HullRound", "HullRound"),
    ]
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Variant & Median loss vs. HR & Median certificate \\",
        r"\midrule",
    ]
    for key, label in variants:
        group = [r for r in ablation if r["variant"] == key]
        lines.append(
            f"{label} & {fmt_pct(median([float(r['loss_vs_hullround']) for r in group]), 3)} & "
            f"{fmt(median([float(r['certificate_value']) for r in group]), 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "extended_ablation_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$n$ & Runs & SCIP opt. & HR time & SCIP time & SCIP inc.-HR \\",
        r"\midrule",
    ]
    for n in sorted({int(r["n"]) for r in large}):
        group = [r for r in large if int(r["n"]) == n]
        cert = sum(1 for r in group if r["scip_certified"] is True or r["scip_certified"] == "True")
        lines.append(
            f"{n} & {len(group)} & {cert}/{len(group)} & "
            f"{fmt(median([float(r['hullround_runtime_s']) for r in group]), 2)} & "
            f"{fmt(median([float(r['scip_runtime_s']) for r in group]), 2)} & "
            f"{fmt_pct(median([float(r['scip_incumbent_minus_hullround_pct']) for r in group]), 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "extended_large_solver_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Rank & Seed & $n$ & $\Gamma$ & $L_{\rm rd}/\Delta V_{\max}$ & LP gap \\",
        r"\midrule",
    ]
    for r in tight[:5]:
        lines.append(
            f"{int(r['rank'])} & {int(r['seed'])} & {int(r['n'])} & {int(r['gamma'])} & "
            f"{fmt(float(r['l_rd_over_delta']), 3)} & {fmt_pct(float(r['gap_lp']), 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "extended_bound_tightness_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_macros(
    ablation: List[Dict[str, object]],
    time_quality: List[Dict[str, object]],
    large: List[Dict[str, object]],
    sensitivity: List[Dict[str, object]],
    stress: List[Dict[str, object]],
    retail_path: List[Dict[str, object]],
    tight: List[Dict[str, object]],
    gap_scaling: List[Dict[str, object]],
    method_frontier: List[Dict[str, object]],
    calibration: List[Dict[str, object]],
) -> None:
    repair = [r for r in ablation if r["variant"] == "repair_no_completion_selected_theta"]
    no_comp = [r for r in ablation if r["variant"] == "global_no_completion"]
    scip_tq = [r for r in time_quality if r["method"] == "SCIP-FullRobust"]
    last_limit = max(float(r["time_limit_s"]) for r in scip_tq) if scip_tq else float("nan")
    last_rows = [r for r in scip_tq if abs(float(r["time_limit_s"]) - last_limit) <= 1e-9]
    n_sens = int(sensitivity[0]["n"]) if sensitivity else 0
    sqrt_gamma = int(math.floor(math.sqrt(n_sens))) if n_sens else 0
    sqrt_sens = [r for r in sensitivity if int(r["gamma"]) == sqrt_gamma]
    full_sens = [r for r in sensitivity if int(r["gamma"]) == int(r["n"])]
    robust_stress = [r for r in stress if int(r["gamma"]) == int(math.floor(math.sqrt(int(r["n"]))))]
    nominal_stress = [r for r in stress if int(r["gamma"]) == 0]
    retail_sqrt = [r for r in retail_path if int(r["gamma"]) == 17]
    large_200 = [r for r in large if int(r["n"]) == 200]
    large_500 = [r for r in large if int(r["n"]) == 500]
    max_gap_n = max(int(r["n"]) for r in gap_scaling) if gap_scaling else 0
    max_gap_rows = [r for r in gap_scaling if int(r["n"]) == max_gap_n]
    method_n = max(int(r["n"]) for r in method_frontier) if method_frontier else 0
    method_sqrt_gamma = int(math.floor(math.sqrt(method_n))) if method_n else 0
    method_sqrt = [r for r in method_frontier if int(r["gamma"]) == method_sqrt_gamma]
    method_nominal = [r for r in method_frontier if str(r["method"]) == "Nominal" and int(r["gamma"]) == method_sqrt_gamma]
    method_hull = [r for r in method_frontier if str(r["method"]) == "HullRound" and int(r["gamma"]) == method_sqrt_gamma]
    method_box = [r for r in method_frontier if str(r["method"]) == "Box robust" and int(r["gamma"]) == method_sqrt_gamma]
    model_alphas = sorted({float(r["model_alpha"]) for r in calibration})
    diag_viol = [
        median([float(r["violation_iid"]) for r in calibration if abs(float(r["model_alpha"]) - a) < 1e-9 and abs(float(r["true_alpha"]) - a) < 1e-9])
        for a in model_alphas
    ]
    macros = {
        "ExtendedAblationRuns": len({(r["seed"], r["gamma"]) for r in ablation}),
        "ExtendedRepairGainMedian": fmt_pct(median([float(r["loss_vs_hullround"]) for r in repair]), 3),
        "ExtendedNoCompletionLossMedian": fmt_pct(median([float(r["loss_vs_hullround"]) for r in no_comp]), 3),
        "ExtendedScipTqCertifiedAtLimit": f"{sum(1 for r in last_rows if r['certified_optimal'] is True or r['certified_optimal'] == 'True')}/{len(last_rows)}",
        "ExtendedScipTqLimit": fmt(last_limit, 0),
        "ExtendedLargeRuns": len(large),
        "ExtendedLargeScipCertified": f"{sum(1 for r in large if r['scip_certified'] is True or r['scip_certified'] == 'True')}/{len(large)}",
        "ExtendedLargeMaxN": max(int(r["n"]) for r in large) if large else 0,
        "ExtendedLargeHrTimeTwoHundred": fmt(median([float(r["hullround_runtime_s"]) for r in large_200]), 2) if large_200 else "n/r",
        "ExtendedLargeScipTimeTwoHundred": fmt(median([float(r["scip_runtime_s"]) for r in large_200]), 2) if large_200 else "n/r",
        "ExtendedLargeHrTimeFiveHundred": fmt(median([float(r["hullround_runtime_s"]) for r in large_500]), 2) if large_500 else "n/r",
        "ExtendedLargeScipTimeFiveHundred": fmt(median([float(r["scip_runtime_s"]) for r in large_500]), 2) if large_500 else "n/r",
        "ExtendedSensitivityCells": len({(r["seed"], r["m"], r["alpha"], r["gamma"]) for r in sensitivity}),
        "ExtendedSensitivitySqrtMinRevenue": fmt(min(float(r["revenue_ratio_vs_gamma0"]) for r in sqrt_sens), 3),
        "ExtendedSensitivityFullMinRevenue": fmt(min(float(r["revenue_ratio_vs_gamma0"]) for r in full_sens), 3),
        "ExtendedSensitivityMinCertificate": fmt(min(float(r["certificate_value"]) for r in sensitivity), 2),
        "ExtendedStressProtocols": len({r["protocol"] for r in stress}),
        "ExtendedStressNominalWorstMedian": fmt_pct(
            max(median([float(r["violation_probability"]) for r in nominal_stress if r["protocol"] == p]) for p in {r["protocol"] for r in stress}),
            2,
        ),
        "ExtendedStressRobustWorstMedian": fmt_pct(
            max(median([float(r["violation_probability"]) for r in robust_stress if r["protocol"] == p]) for p in {r["protocol"] for r in stress}),
            2,
        ),
        "ExtendedRetailSqrtShareChangedMedian": fmt_pct(median([float(r["share_changed"]) for r in retail_sqrt]), 2),
        "ExtendedRetailSqrtAbsChangeMedian": fmt(median([float(r["avg_abs_price_change"]) for r in retail_sqrt]), 3),
        "ExtendedTightestBoundRatio": fmt(max(float(r["l_rd_over_delta"]) for r in tight), 3),
        "ExtendedGapScalingMaxN": str(max_gap_n),
        "ExtendedGapScalingMaxNMedianGap": fmt_pct(median([float(r["gap_lp"]) for r in max_gap_rows]), 3) if max_gap_rows else "n/r",
        "ExtendedMethodFrontierN": str(method_n),
        "ExtendedMethodNominalSqrtViolation": fmt_pct(median([float(r["violation_iid"]) for r in method_nominal]), 2) if method_nominal else "n/r",
        "ExtendedMethodHullSqrtViolation": fmt_pct(median([float(r["violation_iid"]) for r in method_hull]), 2) if method_hull else "n/r",
        "ExtendedMethodBoxSqrtRevenueLoss": fmt_pct(1.0 - median([float(r["revenue_ratio_vs_nominal"]) for r in method_box]), 2) if method_box else "n/r",
        "ExtendedCalibrationDiagonalWorst": fmt_pct(max(diag_viol), 2) if diag_viol else "n/r",
    }
    lines = ["% Auto-generated by scripts/run_extended_publishable_experiments.py"]
    for key, value in macros.items():
        lines.append(f"\\newcommand{{\\{key}}}{{{value}}}")
    EXT_MACROS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a small end-to-end smoke design.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse completed full-design CSVs when their expected row counts are present.")
    parser.add_argument("--solver-time-limit", type=float, default=20.0, help="SCIP limit for larger time-limited solves.")
    parser.add_argument("--stress-scenarios", type=int, default=2000)
    return parser.parse_args()


def read_existing(name: str, expected_rows: int, enabled: bool) -> List[Dict[str, object]] | None:
    path = CSV_DIR / name
    if not enabled or not path.exists():
        return None
    rows = read_csv(path)
    if len(rows) >= expected_rows:
        print(f"[extended-reuse] {name} ({len(rows)} rows)", flush=True)
        return rows
    return None


def main() -> None:
    args = parse_args()
    ensure_dirs()
    t0 = time.perf_counter()
    ablation = read_existing("extended_ablation.csv", 48, args.reuse_existing and not args.smoke) or run_ablation(args.smoke)
    time_quality = read_existing("extended_time_quality.csv", 15, args.reuse_existing and not args.smoke) or run_time_quality(args.smoke, args.solver_time_limit)
    large = read_existing("extended_large_time_limited.csv", 4, args.reuse_existing and not args.smoke) or run_large_time_limited(args.smoke, args.solver_time_limit)
    sensitivity = read_existing("extended_sensitivity_grid.csv", 120, args.reuse_existing and not args.smoke) or run_sensitivity_grid(args.smoke)
    stress = read_existing("extended_out_of_model_stress.csv", 40, args.reuse_existing and not args.smoke) or run_out_of_model_stress(args.smoke, args.stress_scenarios if not args.smoke else 300)
    gap_scaling = read_existing("extended_gap_scaling.csv", 12, args.reuse_existing and not args.smoke) or run_gap_scaling(args.smoke)
    method_frontier = read_existing("extended_method_frontier.csv", 90, args.reuse_existing and not args.smoke) or run_method_frontier(args.smoke, args.stress_scenarios, args.solver_time_limit)
    calibration = read_existing("extended_calibration_grid.csv", 60, args.reuse_existing and not args.smoke) or run_calibration_grid(args.smoke, args.stress_scenarios)
    retail_path = make_retail_decision_path()
    tight = make_bound_tightness()

    plot_ablation(ablation)
    plot_extended_gap_validation(read_csv(CSV_DIR / "synthetic_gap_replicates.csv"), gap_scaling)
    plot_method_frontier(method_frontier)
    plot_time_quality(time_quality)
    plot_large_time_limited(large)
    plot_sensitivity(sensitivity)
    plot_theta_decomposition(sensitivity + large)
    plot_computational_scaling(read_csv(CSV_DIR / "scalability.csv"), sensitivity + large)
    plot_stress(stress)
    plot_robustness_design_diagnostics(sensitivity, stress, calibration)
    plot_retail_path(retail_path)
    plot_retail_decision_diagnostics(
        retail_path,
        read_csv(CSV_DIR / "retail_segment_adjustments.csv"),
        read_csv(CSV_DIR / "retail_margin_concentration.csv"),
    )
    write_tables(ablation, large, tight)
    write_macros(ablation, time_quality, large, sensitivity, stress, retail_path, tight, gap_scaling, method_frontier, calibration)
    print(f"extended publishable experiments complete in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
