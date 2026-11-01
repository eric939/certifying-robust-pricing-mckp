"""Robust pricing of a 200-product portfolio from pricing primitives.

Each product has a reference price, a unit cost, a menu of candidate prices,
a demand estimate at each price and a relative demand deviation. The margin
requirement is  revenue >= kappa * cost of goods sold  when at most Gamma
products take their worst-case demand deviation.
"""
import math

import numpy as np

import robust_mckp as rm

rng = np.random.default_rng(7)
n, m = 200, 12
ref = rng.uniform(10.0, 60.0, n)
cost = ref * rng.uniform(0.55, 0.85, n)
menus = [list(r * np.linspace(0.75, 1.25, m)) for r in ref]          # some prices are inadmissible
elasticity = rng.uniform(0.6, 2.5, n)
demand = [list(rng.uniform(50, 500) * (np.asarray(p) / r) ** -e) for p, r, e in zip(menus, ref, elasticity)]

inst = rm.from_pricing_data(
    reference_prices=ref, unit_costs=cost, weights=np.ones(n),
    price_menus=menus, demands=demand,
    uncertainties=[[0.2] * m for _ in range(n)],          # 20 % relative demand deviation
    margin_target=1.30, tolerances=[0.2] * n,               # admissible: |price - ref| <= 20 % of ref
    gamma=int(math.isqrt(n)),
)
flat = rm.FlatInstance(inst)                                # reuse the arrays for several methods

exact = rm.solve(flat, "exact")
fast = rm.solve(flat, "certified")
print(f"exact      {exact.status:9s} revenue {exact.revenue:12.2f}  {exact.elapsed:.3f}s")
print(f"certified  {fast.status:9s} revenue {fast.revenue:12.2f}  U {fast.upper_bound:12.2f}"
      f"  gap {100 * fast.gap:.2e} %  {fast.elapsed:.3f}s")
print("exactly feasible:", exact.exact.feasible, fast.exact.feasible)

# Original menu IDs: index into the menus passed in, before admissibility filtering.
chosen = [float(menus[i][j]) for i, j in enumerate(exact.original_ids)]
print("first five prices:", [round(p, 2) for p in chosen[:5]])
