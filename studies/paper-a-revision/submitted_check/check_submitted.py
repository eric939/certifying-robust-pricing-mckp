"""Re-check the submitted Paper A evidence (release 2026-08-09-paper-a-final-r2).

(1) exact price admissibility of every option kept by the historical EPS filter;
(2) the reference heuristic's fractional-group recovery (VALIDATION_NOTE.md):
    re-run solve() with a probe and with the corrected choice, and compare.
Read-only with respect to the public repository (runs from a git-archive copy).
"""
import csv, math, sys
from fractions import Fraction
SRC = sys.argv[1]; REL = sys.argv[2]
sys.path[:0] = [SRC, SRC + "/src", SRC + "/scripts"]
import numpy as np
from experiments_nested._common import make_master_portfolio, build_prefix_instance
from robust_mckp import solve
from robust_mckp import rounding
from robust_mckp.utils import EPS

def exact_ok(x, a, s):
    return abs(Fraction(x) - Fraction(a)) <= Fraction(s) * Fraction(a)

designs = []
for seed in range(2001, 2006):
    for n in (12, 20, 30):
        for g in sorted({0, math.isqrt(n), n // 10}): designs.append(("exact", seed, n, 8, g))
for seed in range(1001, 1006):
    for n in (30, 60, 120):
        for g in sorted({0, math.isqrt(n), n // 10}): designs.append(("cert", seed, n, 10, g))
for seed in (4001, 4002, 4003):
    for n in (30, 60, 120, 200):
        for m in (8, 16): designs.append(("scale", seed, n, m, math.isqrt(n)))

# (1) admissibility, per distinct (seed, n, m) prefix
stats = dict(options_kept=0, kept_but_inadmissible=0, rejected_but_admissible=0, near=0)
seen = set()
for (_, seed, n, m, g) in designs:
    if (seed, n, m) in seen: continue
    seen.add((seed, n, m))
    ms = make_master_portfolio(seed=seed, n_max=n, m_max=m, min_admissible_menu=min(8, m))
    a = ms.reference_prices[:n]; u = ms.menu_shocks[:n, :m]; sig = 0.10
    x = a[:, None] * (1.0 + u)
    for i in range(n):
        lo, hi = (1 - sig) * a[i], (1 + sig) * a[i]
        for j in range(m):
            gen_mask = (u[i, j] >= -sig - EPS) and (u[i, j] <= sig + EPS)
            flt = gen_mask and (lo - EPS <= x[i, j] <= hi + EPS)
            ex = exact_ok(float(x[i, j]), float(a[i]), sig)
            stats["options_kept"] += flt
            stats["kept_but_inadmissible"] += flt and not ex
            stats["rejected_but_admissible"] += (not flt) and ex
            stats["near"] += abs(abs(u[i, j]) - sig) < 1e-6
print("ADMISSIBILITY", stats, "prefixes", len(seen))

# (2) rounding probe
orig = rounding._find_fractional_item
log = []
def probe(lp):
    k = orig(lp)
    frac = [i for i, p in enumerate(lp.positions) if 0.0 < p.lambda_ < 1.0]
    if len(frac) > 1 or (k is not None and lp.fractional_item is not None and k != lp.fractional_item):
        log.append((k, lp.fractional_item, [lp.positions[i].lambda_ for i in frac]))
    return k
def corrected(lp):
    if lp.fractional_item is not None and 0.0 < lp.fractional_lambda < 1.0:
        return lp.fractional_item
    return None

ref = {}
with open(REL + "/exact_comparisons.csv") as f:
    for r in csv.DictReader(f): ref[("exact", int(r["seed"]), int(r["n"]), 8, int(r["gamma"]))] = float(r["hullround_objective"])
with open(REL + "/certificate_study.csv") as f:
    for r in csv.DictReader(f): ref[("cert", int(r["seed"]), int(r["n"]), 10, int(r["gamma"]))] = float(r["final_objective"])
with open(REL + "/scalability_study.csv") as f:
    for r in csv.DictReader(f): ref[("scale", int(r["seed"]), int(r["n"]), int(r["m"]), int(r["gamma"]))] = float(r["objective"])

rows = []
for d in designs:
    kind, seed, n, m, g = d
    ms = make_master_portfolio(seed=seed, n_max=n, m_max=m, min_admissible_menu=min(8, m))
    inst = build_prefix_instance(ms, n=n, m=m, alpha=0.10, gamma=g).instance
    log.clear(); rounding._find_fractional_item = probe
    kw = dict(upgrade_completion=True) if kind == "cert" else {}
    h0 = solve(inst, **kw).objective
    nprobe = len(log)
    rounding._find_fractional_item = corrected
    h1 = solve(inst, **kw).objective
    rounding._find_fractional_item = orig
    rows.append((d, ref.get(d), h0, h1, nprobe))
    print(kind, seed, n, m, g, "reproduced" if ref.get(d) == h0 else f"DIFF ref={ref.get(d)} now={h0}",
          "probe_hits", nprobe, "corrected_H", h1, "delta", h1 - h0, flush=True)
print("SUMMARY rows", len(rows), "reproduced", sum(r[1] == r[2] for r in rows),
      "rows_with_spurious_fraction", sum(r[4] > 0 for r in rows),
      "rows_H_changed", sum(r[3] != r[2] for r in rows),
      "max_rel_change", max(abs(r[3] - r[2]) / r[2] for r in rows))
