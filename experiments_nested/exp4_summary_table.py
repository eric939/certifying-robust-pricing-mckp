"""Nested-prefix Experiment 4: manuscript-aligned aggregate summary table."""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List

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
    simulate_adversarial_exact,
    simulate_iid_exact,
    solve_mckp_milp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results_nested")
    parser.add_argument("--tables-dir", type=str, default="tables_nested")
    parser.add_argument("--scenarios", type=int, default=10000)
    parser.add_argument("--enable-milp", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def _gamma_sqrt(n: int) -> int:
    return int(math.floor(math.sqrt(n)))


def _milp_gap_display(milp: Dict[str, object] | None, n: int) -> str:
    if milp is None:
        return "n/a"
    success = bool(milp.get("success", False))
    gap = float(milp.get("mip_gap", np.nan))
    if n <= 200 and success and (not np.isfinite(gap) or gap <= 1e-9):
        return "opt"
    if np.isfinite(gap):
        return f"{gap:.3e}"
    return "n/a"


def main() -> None:
    args = parse_args()

    n_values = [50, 100, 200, 500]
    alpha_values = [0.10, 0.20]
    if args.fast:
        n_values = [50, 100, 200]

    m = 50
    master = make_master_portfolio(n_max=max(n_values), m_max=100, seed=args.seed, sigma=0.10, min_admissible_menu=10)

    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []

    for n in progress(n_values, desc="nested-exp4 n", total=len(n_values)):
        gamma = _gamma_sqrt(n)
        for alpha in progress(alpha_values, desc=f"alpha n={n}", total=len(alpha_values), leave=False):
            prefix0 = build_prefix_instance(master, n=n, m=m, alpha=alpha, gamma=0)
            sol0 = solve(prefix0.instance)
            base_obj = float(sol0.objective)

            prefix = build_prefix_instance(master, n=n, m=m, alpha=alpha, gamma=gamma)

            t0 = time.perf_counter()
            sol = solve(prefix.instance)
            elapsed = time.perf_counter() - t0

            v_list, s_list, t_list = extract_arrays(prefix.instance)
            hulls, s_star_sum = compute_hulls_for_theta(v_list, s_list, t_list, sol.theta)
            capacity = s_star_sum - gamma * sol.theta
            lp_sol = greedy_lp(hulls, capacity)
            rd_val = round_down_value(lp_sol, hulls, v_list)
            l_rd = float(lp_sol.lp_value - rd_val)
            bound = float(delta_v_max(hulls))
            gap_lp = l_rd / lp_sol.lp_value if abs(lp_sol.lp_value) > EPS else np.nan

            gamma_attack_adv = int(min(n, math.floor(1.5 * gamma)))
            viol_adv, _ = simulate_adversarial_exact(
                prefix, sol.selections, gamma_attack=gamma_attack_adv, rng=np.random.default_rng(args.seed + n + int(alpha * 100)), scenarios=args.scenarios
            )
            viol_iid, _ = simulate_iid_exact(
                prefix, sol.selections, rng=np.random.default_rng(args.seed + 10000 + n + int(alpha * 100)), scenarios=args.scenarios
            )

            milp = None
            if args.enable_milp:
                costs_list, _ = build_theta_costs(s_list, t_list, sol.theta)
                milp = solve_mckp_milp(
                    v_list,
                    costs_list,
                    capacity,
                    time_limit=120.0 if n > 75 else 600.0,
                )

            rows.append(
                {
                    "n": n,
                    "m": m,
                    "alpha": alpha,
                    "gamma": gamma,
                    "time_s": elapsed,
                    "l_rd": l_rd,
                    "bound": bound,
                    "gap_lp": gap_lp,
                    "milp_gap": float(milp["mip_gap"]) if milp is not None and np.isfinite(float(milp["mip_gap"])) else np.nan,
                    "milp_gap_display": _milp_gap_display(milp, n),
                    "viol_adv": viol_adv,
                    "viol_iid": viol_iid,
                    "rev_ratio": (float(sol.objective) / base_obj) if base_obj > 0 else np.nan,
                    "theta": float(sol.theta),
                    "certificate_value": float(sol.certificate_value),
                }
            )

    # Save CSV
    csv_path = results_dir / "exp4_summary_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)

    # Save LaTeX rows in the manuscript's column order.
    rows_sorted = sorted(rows, key=lambda r: (int(r["n"]), float(r["alpha"])))
    body_path = tables_dir / "numerics_summary_rows.tex"
    with body_path.open("w") as f:
        for r in rows_sorted:
            f.write(
                f"${int(r['n'])}$ & {float(r['alpha']):.2f} & ${int(r['gamma'])}$ & "
                f"{float(r['time_s']):.3f} & {float(r['l_rd']):.6g} & {float(r['bound']):.6g} & "
                f"{float(r['gap_lp']):.3e} & {float(r['rev_ratio']):.3f} & "
                f"{float(r['viol_adv']):.3e} & {float(r['viol_iid']):.3e} \\\\\n"
            )

    tabular_path = tables_dir / "numerics_summary.tex"
    with tabular_path.open("w") as f:
        f.write("% Auto-generated nested-prefix summary table\n")
        f.write("\\begin{tabular}{r r r r r r r r r r}\n")
        f.write("\\toprule\n")
        f.write("$n$ & $\\alpha$ & $\\Gamma$ & Time (s) & $L_{\\mathrm{rd}}$ & $\\Delta V_{\\max}^\\theta$ & $\\mathrm{Gap}_{\\mathrm{LP}}$ & Rev.\\ ratio & Viol. (adv.) & Viol. (iid) \\\\\n")
        f.write("\\midrule\n")
        for r in rows_sorted:
            f.write(
                f"{int(r['n'])} & {float(r['alpha']):.2f} & {int(r['gamma'])} & "
                f"{float(r['time_s']):.3f} & {float(r['l_rd']):.6g} & {float(r['bound']):.6g} & "
                f"{float(r['gap_lp']):.3e} & {float(r['rev_ratio']):.3f} & "
                f"{float(r['viol_adv']):.3e} & {float(r['viol_iid']):.3e} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"Summary CSV saved to {csv_path}")
    print(f"LaTeX rows saved to {body_path}")
    print(f"LaTeX tabular saved to {tabular_path}")


if __name__ == "__main__":
    main()
