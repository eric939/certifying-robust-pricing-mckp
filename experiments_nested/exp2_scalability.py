"""Nested-prefix Experiment 2: scalability, hull compression, and theta-efficiency."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from robust_mckp import Option, PricingInstance, solve
from robust_mckp.certificate import compute_certificate
from robust_mckp.greedy import greedy_lp
from robust_mckp.rounding import round_lp_solution
from robust_mckp.solver import (
    _advance_events_to_theta,
    _build_hulls_exact,
    _build_reduced_candidates,
    _compute_item_baselines,
    _extract_arrays,
    _initialize_sweep_states,
    _make_event_arrays,
)
from robust_mckp.utils import EPS

from _common import (
    build_prefix_instance,
    build_theta_costs,
    compute_hulls_for_theta,
    extract_arrays,
    hull_sizes_for_theta,
    make_master_portfolio,
    progress,
    solve_lp_highs_full,
    try_import_matplotlib,
)

# ---------------------------------------------------------------------------
# Publication plot style
# ---------------------------------------------------------------------------
def apply_pub_style() -> None:
    """Apply consistent publication-quality matplotlib defaults."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.4,
            "lines.markersize": 6,
        }
    )


# Consistent palette across all experiment figures
COLORS = {
    "m10": "#1f77b4",
    "m50": "#ff7f0e",
    # Stacked-bar phases
    "preprocessing": "#4c78a8",
    "theta-overhead": "#f58518",
    "hull": "#e45756",
    "greedy": "#72b7b2",
    "round": "#54a24b",
    "cert": "#b279a2",
}
MARKERS = {"m10": "o", "m50": "s"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="figs_nested")
    parser.add_argument("--results-dir", type=str, default="results_nested")
    parser.add_argument("--validate-lp", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--check-public-solve", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fractional_item_count(lp_sol) -> int:
    cnt = 0
    for pos in lp_sol.positions:
        if EPS < pos.lambda_ < 1.0 - EPS:
            cnt += 1
    return cnt


def solve_profiled_exact(instance: PricingInstance) -> Dict[str, object]:
    """Profile the exact optimized solver path phase-by-phase.

    This mirrors `robust_mckp.solver.solve` using its private helpers so that
    the runtime breakdown reflects the same algorithmic logic.
    """

    t_total0 = time.perf_counter()
    t_pre = t_overhead = t_hull = t_greedy = t_round = t_cert = 0.0

    # --- Preprocessing ---
    t0 = time.perf_counter()
    v_list, s_list, t_list = _extract_arrays(instance)
    candidates, raw_count, reduced_count, abs_t_list = _build_reduced_candidates(
        v_list, s_list, t_list
    )
    (
        states,
        inactive_best_vals,
        inactive_best_indices,
        active_best_bases,
        active_best_indices,
        j_star,
        s_star_vals,
        touched_mask,
    ) = _initialize_sweep_states(v_list, s_list, abs_t_list)
    event_values, event_items, event_options = _make_event_arrays(abs_t_list)
    event_ptr = 0
    while event_ptr < event_values.size and event_values[event_ptr] <= 0.0 + EPS:
        event_ptr += 1
    _compute_item_baselines(
        theta=0.0,
        inactive_best_vals=inactive_best_vals,
        inactive_best_indices=inactive_best_indices,
        active_best_bases=active_best_bases,
        active_best_indices=active_best_indices,
        j_star_out=j_star,
        s_star_out=s_star_vals,
    )
    prev_j_star = j_star.copy()
    t_pre += time.perf_counter() - t0

    # --- Theta loop ---
    best_obj = -np.inf
    best_lp = -np.inf
    best_theta = float("nan")
    best_sel: Optional[List[int]] = None
    best_cert = float("nan")

    theta_skipped = 0
    theta_cert_infeasible = 0
    theta_round_failed = 0
    improve_count = 0
    hull_rebuilds_per_theta: List[int] = []
    baseline_changes_per_theta: List[int] = []

    for step_idx, theta in enumerate(candidates):
        t0 = time.perf_counter()
        if step_idx > 0:
            event_ptr, _ = _advance_events_to_theta(
                theta=float(theta),
                event_values=event_values,
                event_items=event_items,
                event_options=event_options,
                event_ptr=event_ptr,
                states=states,
                inactive_best_vals=inactive_best_vals,
                inactive_best_indices=inactive_best_indices,
                active_best_bases=active_best_bases,
                active_best_indices=active_best_indices,
                touched_mask=touched_mask,
            )
            _compute_item_baselines(
                theta=float(theta),
                inactive_best_vals=inactive_best_vals,
                inactive_best_indices=inactive_best_indices,
                active_best_bases=active_best_bases,
                active_best_indices=active_best_indices,
                j_star_out=j_star,
                s_star_out=s_star_vals,
            )

        changed_baselines = int(np.count_nonzero(j_star != prev_j_star))
        baseline_changes_per_theta.append(changed_baselines)
        prev_j_star[:] = j_star

        s_star_sum = float(np.sum(s_star_vals))
        capacity = s_star_sum - instance.gamma * float(theta)
        t_overhead += time.perf_counter() - t0

        if capacity < -EPS:
            theta_skipped += 1
            hull_rebuilds_per_theta.append(0)
            continue

        t0 = time.perf_counter()
        hulls = _build_hulls_exact(
            states=states, theta=float(theta), s_star_vals=s_star_vals
        )
        t_hull += time.perf_counter() - t0
        hull_rebuilds_per_theta.append(instance.n_items)

        t0 = time.perf_counter()
        lp_sol = greedy_lp(hulls, capacity)
        t_greedy += time.perf_counter() - t0

        t0 = time.perf_counter()
        discrete = round_lp_solution(
            lp_sol, hulls, capacity, upgrade_completion=True
        )
        t_round += time.perf_counter() - t0
        if discrete is None:
            theta_round_failed += 1
            continue

        selections = [
            int(hulls[i].option_indices[idx])
            for i, idx in enumerate(discrete.vertices)
        ]

        t0 = time.perf_counter()
        cert = compute_certificate(instance, selections)
        t_cert += time.perf_counter() - t0
        if cert < -EPS:
            theta_cert_infeasible += 1
            continue

        obj = float(sum(v_list[i][sel] for i, sel in enumerate(selections)))
        if obj > best_obj + EPS:
            best_obj = obj
            best_lp = float(lp_sol.lp_value)
            best_theta = float(theta)
            best_sel = selections
            best_cert = float(cert)
            improve_count += 1

    total = time.perf_counter() - t_total0

    # Aggregate incremental-sweep statistics
    evaluated_changes = [
        c for c, h in zip(baseline_changes_per_theta, hull_rebuilds_per_theta) if h > 0
    ]

    return {
        "objective": best_obj,
        "lp_value": best_lp,
        "theta": best_theta,
        "selections": best_sel,
        "certificate_value": best_cert,
        "total_time": total,
        "t_preprocess": t_pre,
        "t_theta_overhead": t_overhead,
        "t_hull": t_hull,
        "t_greedy": t_greedy,
        "t_round": t_round,
        "t_cert": t_cert,
        "candidate_count_raw": int(raw_count),
        "candidate_count_reduced": int(reduced_count),
        "theta_evaluated_count": int(candidates.size),
        "theta_skipped_capacity_count": int(theta_skipped),
        "theta_rounding_failed_count": int(theta_round_failed),
        "theta_certificate_infeasible_count": int(theta_cert_infeasible),
        "improve_count": int(improve_count),
        "total_hull_rebuilds": int(sum(hull_rebuilds_per_theta)),
        "hull_rebuilds_per_theta": hull_rebuilds_per_theta,
        "baseline_changes_per_theta": baseline_changes_per_theta,
        "baseline_changes_mean": float(np.mean(evaluated_changes)) if evaluated_changes else 0.0,
        "baseline_changes_median": float(np.median(evaluated_changes)) if evaluated_changes else 0.0,
        "baseline_changes_max": int(max(evaluated_changes)) if evaluated_changes else 0,
    }


# ---------------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------------
def _save_csv(path: Path, records: List[Dict[str, object]]) -> None:
    if not records:
        return
    all_keys = list(records[0].keys())
    for r in records[1:]:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_runtime(runtime_rows, m_values, n_values, output_dir):
    """Panel (a): Runtime vs n on log-log axes."""
    plt = try_import_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    for m in m_values:
        key = f"m{m}"
        rows_m = sorted(
            [r for r in runtime_rows if int(r["m"]) == m], key=lambda r: int(r["n"])
        )
        if not rows_m:
            continue
        ns = np.array([r["n"] for r in rows_m], dtype=float)
        ts = np.array([r["time_s"] for r in rows_m], dtype=float)
        ax.plot(ns, ts, marker=MARKERS[key], color=COLORS[key], label=f"$m = {m}$")

    # Reference slopes anchored at geometric mean of the two m-curves
    n_min, n_max = min(n_values), max(n_values)
    ref_ns = np.array([n_min, n_max], dtype=float)
    # Anchor at the midpoint m-curve value at n_min
    mid_times = []
    for m in m_values:
        row = next((r for r in runtime_rows if int(r["m"]) == m and int(r["n"]) == n_min), None)
        if row:
            mid_times.append(float(row["time_s"]))
    if mid_times:
        t_anchor = np.exp(np.mean(np.log(np.array(mid_times) + 1e-12)))
        ax.plot(
            ref_ns,
            t_anchor * (ref_ns / ref_ns[0]) ** 1.0,
            "k--",
            alpha=0.45,
            linewidth=1.0,
            label="slope 1",
        )
        ax.plot(
            ref_ns,
            t_anchor * (ref_ns / ref_ns[0]) ** 2.0,
            "k:",
            alpha=0.45,
            linewidth=1.0,
            label="slope 2",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(r"Runtime vs. prefix size")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_vs_n.pdf")
    plt.close(fig)


def plot_hull_boxplot(hull_rows_theta0, m_values_hull, n_hull, output_dir):
    """Panel (b): Hull-size compression boxplot at theta=0."""
    plt = try_import_matplotlib()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    data = []
    labels = []
    medians = []
    for m in m_values_hull:
        vals = [int(r["hull"]) for r in hull_rows_theta0 if int(r["m"]) == m]
        if vals:
            data.append(vals)
            labels.append(str(m))
            medians.append(np.median(vals))

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
        medianprops=dict(color="#e45756", linewidth=1.5),
        boxprops=dict(linewidth=1.1),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        patch_artist=True,
    )
    # Light fill for boxes
    box_colors = ["#c6dbef", "#fdd0a2", "#c7e9c0", "#dadaeb"]
    for patch, color in zip(bp["boxes"], box_colors[: len(data)]):
        patch.set_facecolor(color)

    # Annotate medians
    for i, med in enumerate(medians):
        ax.text(
            i + 1,
            med + 0.3,
            f"{med:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#e45756",
        )

    # Diagonal reference: identity line (hull = raw)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    for i, m in enumerate(m_values_hull):
        if str(m) in labels:
            idx = labels.index(str(m))
            ax.hlines(
                m, idx + 0.6, idx + 1.4, colors="gray", linestyles="--", alpha=0.5, linewidth=0.8
            )
            if i == len(m_values_hull) - 1:
                ax.text(idx + 1.42, m, f"$m$={m}", fontsize=7, va="center", color="gray", alpha=0.7)

    ax.set_xlabel("Raw menu size $m$")
    ax.set_ylabel(r"Hull size $|\mathcal{H}_i^\theta|$")
    ax.set_title(r"Hull compression at $\theta=0$ ($n=%d$)" % n_hull)
    fig.tight_layout()
    fig.savefig(output_dir / "hull_sizes.pdf")
    plt.close(fig)


def plot_runtime_breakdown(runtime_rows, n_values, output_dir):
    """Panel (c): Stacked bar chart of runtime by phase at m=50."""
    plt = try_import_matplotlib()
    if plt is None:
        return

    rows_m50 = sorted(
        [r for r in runtime_rows if int(r["m"]) == 50], key=lambda r: int(r["n"])
    )
    if not rows_m50:
        return

    ns = [int(r["n"]) for r in rows_m50]
    x_pos = np.arange(len(ns))
    bar_width = 0.65

    phase_order = ["preprocessing", "theta-overhead", "hull", "greedy", "round", "cert"]
    phase_labels = {
        "preprocessing": "Preprocess",
        "theta-overhead": r"$\theta$-overhead",
        "hull": "Hull build",
        "greedy": "Greedy LP",
        "round": "Rounding",
        "cert": "Certificate",
    }

    # Column names in runtime_rows use `t_preprocess`, while the display label is
    # "preprocessing". Support both spellings so plotting also works with older CSVs.
    csv_key_map = {
        "preprocessing": ("t_preprocess", "t_preprocessing"),
        "theta-overhead": ("t_theta_overhead",),
        "hull": ("t_hull",),
        "greedy": ("t_greedy",),
        "round": ("t_round",),
        "cert": ("t_cert",),
    }

    stacks = {}
    for key in phase_order:
        candidates = csv_key_map[key]
        chosen_key = None
        for ck in candidates:
            if ck in rows_m50[0]:
                chosen_key = ck
                break
        if chosen_key is None:
            raise KeyError(f"Missing runtime breakdown column for phase '{key}'. Tried: {candidates}")
        stacks[key] = np.array([float(r[chosen_key]) for r in rows_m50])

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bottom = np.zeros(len(ns))
    for key in phase_order:
        vals = stacks[key]
        ax.bar(
            x_pos,
            vals,
            bar_width,
            bottom=bottom,
            color=COLORS[key],
            edgecolor="white",
            linewidth=0.4,
            label=phase_labels[key],
        )
        bottom += vals

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(n) for n in ns], rotation=0)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("Time (s)")
    ax.set_title(r"Runtime breakdown ($m=50$)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_breakdown.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    n_values = [30, 50, 75, 100, 150, 200, 300, 500]
    m_values_runtime = [10, 50]
    m_values_hull = [10, 20, 50, 100]
    if args.fast:
        n_values = [30, 50, 75, 100, 150, 200]

    alpha = 0.10
    master = make_master_portfolio(
        n_max=max(n_values),
        m_max=max(m_values_hull),
        seed=args.seed,
        sigma=0.10,
        min_admissible_menu=min(m_values_hull),
    )

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    runtime_rows: List[Dict[str, object]] = []
    hull_rows_theta0: List[Dict[str, object]] = []
    hull_rows_best: List[Dict[str, object]] = []
    theta_eff_rows: List[Dict[str, object]] = []
    box_rows: List[Dict[str, object]] = []
    lp_val_rows: List[Dict[str, object]] = []

    # -----------------------------------------------------------------------
    # Main runtime + profiling loop: m × n grid
    # -----------------------------------------------------------------------
    tasks = [(m, n) for m in m_values_runtime for n in n_values]
    for m, n in progress(tasks, desc="nested-exp2", total=len(tasks)):
        gamma = int(math.floor(math.sqrt(n)))
        prefix = build_prefix_instance(master, n=n, m=m, alpha=alpha, gamma=gamma)

        prof = solve_profiled_exact(prefix.instance)

        if args.check_public_solve:
            sol_pub = solve(prefix.instance)
            if (
                abs(sol_pub.objective - float(prof["objective"])) > 1e-9
                or abs(sol_pub.theta - float(prof["theta"])) > 1e-9
            ):
                raise RuntimeError(f"Profiled solver mismatch at (n={n}, m={m})")

        runtime_rows.append(
            {
                "n": n,
                "m": m,
                "alpha": alpha,
                "gamma": gamma,
                "time_s": prof["total_time"],
                "t_preprocess": prof["t_preprocess"],
                "t_theta_overhead": prof["t_theta_overhead"],
                "t_hull": prof["t_hull"],
                "t_greedy": prof["t_greedy"],
                "t_round": prof["t_round"],
                "t_cert": prof["t_cert"],
                "objective": prof["objective"],
                "theta": prof["theta"],
                "candidate_count_raw": prof["candidate_count_raw"],
                "candidate_count_reduced": prof["candidate_count_reduced"],
                "theta_evaluated_count": prof["theta_evaluated_count"],
                "theta_skipped_capacity_count": prof["theta_skipped_capacity_count"],
                "improve_count": prof["improve_count"],
                "total_hull_rebuilds": prof["total_hull_rebuilds"],
                "baseline_changes_mean": prof["baseline_changes_mean"],
                "baseline_changes_median": prof["baseline_changes_median"],
                "baseline_changes_max": prof["baseline_changes_max"],
            }
        )

        theta_eff_rows.append(
            {
                "n": n,
                "m": m,
                "gamma": gamma,
                "candidate_count_raw": prof["candidate_count_raw"],
                "candidate_count_reduced": prof["candidate_count_reduced"],
                "candidate_reduction_ratio": (
                    float(prof["candidate_count_reduced"])
                    / float(prof["candidate_count_raw"])
                    if float(prof["candidate_count_raw"]) > 0
                    else np.nan
                ),
                "theta_evaluated_count": prof["theta_evaluated_count"],
                "theta_skipped_capacity_count": prof["theta_skipped_capacity_count"],
                "theta_skip_rate": (
                    float(prof["theta_skipped_capacity_count"])
                    / float(prof["theta_evaluated_count"])
                    if float(prof["theta_evaluated_count"]) > 0
                    else np.nan
                ),
                "improve_count": prof["improve_count"],
                "theta_improve_rate": (
                    float(prof["improve_count"])
                    / float(prof["theta_evaluated_count"])
                    if float(prof["theta_evaluated_count"]) > 0
                    else np.nan
                ),
                "baseline_changes_mean": prof["baseline_changes_mean"],
                "baseline_changes_median": prof["baseline_changes_median"],
                "baseline_changes_max": prof["baseline_changes_max"],
            }
        )

        # Hull sizes at optimal theta
        v_list, s_list, t_list = extract_arrays(prefix.instance)
        if np.isfinite(float(prof["theta"])):
            raw_sizes, undom_sizes, hull_sizes = hull_sizes_for_theta(
                v_list, s_list, t_list, float(prof["theta"])
            )
            for raw, und, hull in zip(raw_sizes, undom_sizes, hull_sizes):
                hull_rows_best.append(
                    {
                        "n": n,
                        "m": m,
                        "theta": prof["theta"],
                        "raw": raw,
                        "undominated": und,
                        "hull": hull,
                    }
                )

        # Optional LP validation
        if args.validate_lp:
            abs_t = np.concatenate([np.abs(t) for t in t_list])
            candidates = np.unique(
                np.concatenate([np.array([0.0], dtype=float), abs_t])
            )
            thetas = sorted(
                set([0.0, float(candidates[-1]), float(prof["theta"])])
            )
            for theta in thetas:
                costs_list, s_star_sum = build_theta_costs(s_list, t_list, theta)
                cap = s_star_sum - gamma * theta
                if cap < -EPS:
                    continue
                hulls, _ = compute_hulls_for_theta(v_list, s_list, t_list, theta)
                lp_greedy = greedy_lp(hulls, cap)
                lp_highs = solve_lp_highs_full(v_list, costs_list, cap)
                if lp_highs is None:
                    continue
                lp_val_rows.append(
                    {
                        "n": n,
                        "m": m,
                        "gamma": gamma,
                        "theta": theta,
                        "lp_greedy": float(lp_greedy.lp_value),
                        "lp_highs": float(lp_highs["objective"]),
                        "abs_residual": abs(
                            float(lp_greedy.lp_value) - float(lp_highs["objective"])
                        ),
                        "fractional_item_count": _fractional_item_count(lp_greedy),
                        "knapsack_dual_highs": lp_highs.get(
                            "knapsack_dual_max_form", np.nan
                        ),
                    }
                )

    # -----------------------------------------------------------------------
    # Hull compression study: n=max, theta=0, m in {10,20,50,100}
    # -----------------------------------------------------------------------
    n_hull = max(n_values)
    for m in progress(
        m_values_hull, desc="hull-compression", total=len(m_values_hull), leave=False
    ):
        prefix = build_prefix_instance(master, n=n_hull, m=m, alpha=alpha, gamma=0)
        v_list, s_list, t_list = extract_arrays(prefix.instance)
        raw_sizes, undom_sizes, hull_sizes = hull_sizes_for_theta(
            v_list, s_list, t_list, theta=0.0
        )
        for raw, und, hull in zip(raw_sizes, undom_sizes, hull_sizes):
            hull_rows_theta0.append(
                {"n": n_hull, "m": m, "raw": raw, "undominated": und, "hull": hull}
            )

    # -----------------------------------------------------------------------
    # Box-case cross-check (Gamma=n) for m=50
    # -----------------------------------------------------------------------
    for n in progress(
        n_values, desc="box-check", total=len(n_values), leave=False
    ):
        prefix = build_prefix_instance(master, n=n, m=50, alpha=alpha, gamma=n)
        sol_gamma = solve(prefix.instance)

        v_list, s_list, t_list = extract_arrays(prefix.instance)
        box_items = []
        for v, s, t in zip(v_list, s_list, t_list):
            box_items.append(
                [
                    Option(float(vj), float(sj - abs(tj)), 0.0)
                    for vj, sj, tj in zip(v, s, t)
                ]
            )
        inst_box = PricingInstance(items=box_items, gamma=0)
        sol_box = solve(inst_box)
        diff = abs(sol_gamma.objective - sol_box.objective)
        box_rows.append(
            {
                "n": n,
                "m": 50,
                "objective_gamma_n": sol_gamma.objective,
                "objective_box": sol_box.objective,
                "abs_diff": diff,
                "rel_diff": (
                    diff / sol_box.objective
                    if abs(sol_box.objective) > EPS
                    else np.nan
                ),
            }
        )

    # -----------------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------------
    _save_csv(results_dir / "exp2_runtime.csv", runtime_rows)
    _save_csv(results_dir / "exp2_theta_efficiency.csv", theta_eff_rows)
    _save_csv(results_dir / "exp2_hull_sizes_theta0.csv", hull_rows_theta0)
    _save_csv(results_dir / "exp2_hull_sizes_best_theta.csv", hull_rows_best)
    _save_csv(results_dir / "exp2_box_check.csv", box_rows)
    _save_csv(results_dir / "exp2_lp_validation.csv", lp_val_rows)

    print(f"\nResults saved to {results_dir}/")
    print(f"  exp2_runtime.csv            ({len(runtime_rows)} rows)")
    print(f"  exp2_theta_efficiency.csv   ({len(theta_eff_rows)} rows)")
    print(f"  exp2_hull_sizes_theta0.csv  ({len(hull_rows_theta0)} rows)")
    print(f"  exp2_hull_sizes_best_theta.csv ({len(hull_rows_best)} rows)")
    print(f"  exp2_box_check.csv          ({len(box_rows)} rows)")
    if lp_val_rows:
        print(f"  exp2_lp_validation.csv      ({len(lp_val_rows)} rows)")

    # Print box-check summary
    if box_rows:
        max_diff = max(float(r["abs_diff"]) for r in box_rows)
        max_rel = max(float(r["rel_diff"]) for r in box_rows if np.isfinite(float(r["rel_diff"])))
        print(
            f"\nBox-case cross-check: max |obj_gamma_n - obj_box| = {max_diff:.2e} "
            f"(max relative {max_rel:.2e})"
        )

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    plt = try_import_matplotlib()
    if plt is None:
        print("matplotlib not available; plots skipped")
        return

    apply_pub_style()
    plot_runtime(runtime_rows, m_values_runtime, n_values, output_dir)
    plot_hull_boxplot(hull_rows_theta0, m_values_hull, n_hull, output_dir)
    plot_runtime_breakdown(runtime_rows, n_values, output_dir)

    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
