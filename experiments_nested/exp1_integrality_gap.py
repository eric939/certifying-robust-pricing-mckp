"""Nested-prefix Experiment 1: additive rounding-loss bound and implied O(1/n) decay."""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from robust_mckp import solve
from robust_mckp.greedy import greedy_lp
from robust_mckp.utils import EPS

from _common import (
    build_prefix_instance,
    build_theta_costs,
    compute_hulls_for_theta,
    delta_v_max,
    extract_arrays,
    make_master_portfolio,
    progress,
    round_down_value,
    solve_mckp_milp,
    try_import_matplotlib,
)


GAMMA_REGIMES = [
    ("0", lambda n: 0),
    ("sqrt(n)", lambda n: int(math.floor(math.sqrt(n)))),
    ("0.1n", lambda n: int(math.floor(0.1 * n))),
]

REGIME_DISPLAY = {
    "0": r"$\Gamma = 0$",
    "sqrt(n)": r"$\Gamma = \lfloor\sqrt{n}\rfloor$",
    "0.1n": r"$\Gamma = \lfloor 0.1n \rfloor$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Master portfolio seed")
    parser.add_argument("--output-dir", type=str, default="figs_nested")
    parser.add_argument("--results-dir", type=str, default="results_nested")
    parser.add_argument("--enable-milp", action="store_true", help="Run fixed-theta MILP for n<=200")
    parser.add_argument("--global-milp", action="store_true", help="Run global MILP benchmark for n<=75")
    parser.add_argument("--fast", action="store_true", help="Smaller n-grid for quick checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n_values = [30, 50, 75, 100, 150, 200, 300, 500]
    if args.fast:
        n_values = [30, 50, 75, 100, 150, 200]

    m = 50
    alpha = 0.10
    master = make_master_portfolio(
        n_max=max(n_values),
        m_max=100,
        seed=args.seed,
        sigma=0.10,
        min_admissible_menu=10,
    )

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    global_rows: List[Dict[str, object]] = []

    tasks = [(n, label, gamma_fn(n)) for n in n_values for label, gamma_fn in GAMMA_REGIMES]
    for n, regime_label, gamma in progress(tasks, desc="nested-exp1", total=len(tasks)):
        prefix = build_prefix_instance(master, n=n, m=m, alpha=alpha, gamma=gamma)

        t0 = time.perf_counter()
        sol = solve(prefix.instance)
        runtime = time.perf_counter() - t0

        v_list, s_list, t_list = extract_arrays(prefix.instance)
        hulls, s_star_sum = compute_hulls_for_theta(v_list, s_list, t_list, sol.theta)
        capacity = s_star_sum - gamma * sol.theta
        lp_sol = greedy_lp(hulls, capacity)
        rd_val = round_down_value(lp_sol, hulls, v_list)
        l_rd = float(lp_sol.lp_value - rd_val)
        bound = float(delta_v_max(hulls))
        gap_lp_rd = l_rd / lp_sol.lp_value if abs(lp_sol.lp_value) > EPS else float("nan")
        if bound > EPS:
            ratio_lrd_bound = l_rd / bound
        elif abs(l_rd) <= EPS:
            ratio_lrd_bound = 0.0
        else:
            ratio_lrd_bound = float("nan")
        scaled_gap_n = n * gap_lp_rd if np.isfinite(gap_lp_rd) else float("nan")

        instr = (sol.metadata or {}).get("instrumentation", {})
        rec: Dict[str, object] = {
            "n": n,
            "m": m,
            "alpha": alpha,
            "gamma": gamma,
            "gamma_regime": regime_label,
            "theta": float(sol.theta),
            "time_s": runtime,
            "objective": float(sol.objective),
            "lp_value_selected_theta": float(lp_sol.lp_value),
            "l_rd": l_rd,
            "bound": bound,
            "gap_lp_rd": gap_lp_rd,
            "ratio_lrd_bound": ratio_lrd_bound,
            "n_gap_lp": scaled_gap_n,
            "certificate_value": float(sol.certificate_value),
            "candidate_count_raw": instr.get("candidate_count_raw", np.nan),
            "candidate_count_reduced": instr.get("candidate_count_reduced", np.nan),
            "theta_skipped_capacity_count": instr.get("theta_skipped_capacity_count", np.nan),
            "theta_evaluated_count": instr.get("theta_evaluated_count", np.nan),
        }

        if args.enable_milp and n <= 200:
            costs_list, _ = build_theta_costs(s_list, t_list, sol.theta)
            milp = solve_mckp_milp(v_list, costs_list, capacity, time_limit=120.0)
            if milp is not None:
                rec["milp_obj"] = milp["objective"]
                rec["milp_gap"] = milp["mip_gap"]
                rec["milp_success"] = milp["success"]
                rec["milp_status"] = milp["status"]
                rec["milp_message"] = milp["message"]
                obj = float(milp["objective"])
                if np.isfinite(obj) and abs(obj) > EPS:
                    rec["gap_ip"] = (obj - sol.objective) / obj

        if args.global_milp and n <= 75:
            abs_t = np.concatenate([np.abs(t) for t in t_list])
            candidates = np.unique(np.concatenate([np.array([0.0], dtype=float), abs_t]))
            best_global = -np.inf
            best_theta = np.nan
            for theta_cand in progress(
                candidates,
                desc=f"global-milp n={n} {regime_label}",
                total=int(candidates.size),
                leave=False,
            ):
                costs_list, s_sum = build_theta_costs(s_list, t_list, float(theta_cand))
                cap = s_sum - gamma * float(theta_cand)
                if cap < -EPS:
                    continue
                milp = solve_mckp_milp(v_list, costs_list, cap, time_limit=600.0)
                if milp is None:
                    continue
                obj = float(milp["objective"])
                if not np.isfinite(obj):
                    continue
                if obj > best_global + EPS:
                    best_global = obj
                    best_theta = float(theta_cand)
            if np.isfinite(best_global):
                global_rows.append(
                    {
                        "n": n,
                        "gamma": gamma,
                        "gamma_regime": regime_label,
                        "global_obj": best_global,
                        "global_theta": best_theta,
                        "alg_obj": float(sol.objective),
                        "gap_global": (best_global - sol.objective) / best_global if abs(best_global) > EPS else np.nan,
                    }
                )

        rows.append(rec)

    # Save raw rows
    raw_path = results_dir / "exp1_gap.csv"
    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)

    if global_rows:
        global_path = results_dir / "exp1_gap_global_milp.csv"
        with global_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in global_rows))))
            writer.writeheader()
            writer.writerows(global_rows)

    # Simple summary (deterministic one-row per configuration)
    summary_path = results_dir / "exp1_gap_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n",
                "gamma",
                "gamma_regime",
                "l_rd",
                "bound",
                "gap_lp_rd",
                "ratio_lrd_bound",
                "n_gap_lp",
                "time_s",
                "theta",
                "candidate_count_raw",
                "candidate_count_reduced",
            ],
        )
        writer.writeheader()
        for r in sorted(rows, key=lambda rr: (int(rr["n"]), str(rr["gamma_regime"]))):
            writer.writerow(
                {
                    "n": r["n"],
                    "gamma": r["gamma"],
                    "gamma_regime": r["gamma_regime"],
                    "l_rd": r["l_rd"],
                    "bound": r["bound"],
                    "gap_lp_rd": r["gap_lp_rd"],
                    "ratio_lrd_bound": r["ratio_lrd_bound"],
                    "n_gap_lp": r["n_gap_lp"],
                    "time_s": r["time_s"],
                    "theta": r["theta"],
                    "candidate_count_raw": r["candidate_count_raw"],
                    "candidate_count_reduced": r["candidate_count_reduced"],
                }
            )

    plt = try_import_matplotlib()
    if plt is None:
        print("matplotlib not available; plots skipped")
        print(f"Results saved to {results_dir}/")
        return

    colors = {"0": "#1f77b4", "sqrt(n)": "#ff7f0e", "0.1n": "#2ca02c"}
    markers = {"0": "o", "sqrt(n)": "s", "0.1n": "D"}

    # (a) Ratio L_rd / DeltaVmax vs n (direct test of additive bound slack)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for regime_label, _ in GAMMA_REGIMES:
        regime_rows = sorted([r for r in rows if r["gamma_regime"] == regime_label], key=lambda r: int(r["n"]))
        ns = np.array([r["n"] for r in regime_rows], dtype=float)
        ratio = np.array([r["ratio_lrd_bound"] for r in regime_rows], dtype=float)
        color = colors[regime_label]
        marker = markers[regime_label]

        ax.plot(
            ns,
            ratio,
            marker=marker,
            linestyle="-",
            color=color,
            markersize=6,
            label=REGIME_DISPLAY[regime_label],
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8, label=r"Bound threshold $=1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$L_{\mathrm{rd}} / \Delta V_{\max}^{\theta}$")
    ax.set_title(r"Normalized round-down loss vs $n$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(output_dir / "gap_loss_vs_n.pdf")
    plt.close(fig)

    # (b) n * Gap_LP vs n (direct test of O(1/n) scaling)
    fig2, ax2 = plt.subplots(figsize=(5.1, 4.2))
    for regime_label, _ in GAMMA_REGIMES:
        pts = [r for r in rows if r["gamma_regime"] == regime_label]
        pts = sorted(pts, key=lambda r: int(r["n"]))
        ns = np.array([r["n"] for r in pts], dtype=float)
        scaled = np.array([r["n_gap_lp"] for r in pts], dtype=float)
        ax2.plot(
            ns,
            scaled,
            color=colors[regime_label],
            marker=markers[regime_label],
            linewidth=1.5,
            markersize=6,
            alpha=0.9,
            label=REGIME_DISPLAY[regime_label],
        )
    all_scaled = np.array([float(r["n_gap_lp"]) for r in rows], dtype=float)
    all_scaled = all_scaled[np.isfinite(all_scaled)]
    if all_scaled.size:
        y_med = float(np.median(all_scaled))
        ax2.axhline(
            y_med,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=r"Median of $n \times \mathrm{Gap}_{\mathrm{LP}}$",
        )
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$n$")
    ax2.set_ylabel(r"$n \times \mathrm{Gap}_{\mathrm{LP}}$")
    ax2.set_title(r"Scaled relative gap vs $n$")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(output_dir / "loss_vs_bound.pdf")
    plt.close(fig2)

    print(f"Results saved to {results_dir}/")
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
