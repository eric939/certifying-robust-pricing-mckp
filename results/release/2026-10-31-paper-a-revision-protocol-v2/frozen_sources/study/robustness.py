"""Protocol v2, item 3: price of robustness and out-of-sample margin violations (Panel S).

For every Panel S configuration (base portfolio, Gamma, radius) the decision is
the one returned by the threshold DnC in protocol v1 (optimal within relative
tolerance 1e-6), read from the v1 evidence by its original menu IDs.  The
nominal decision is the DnC decision of the same base at Gamma = 0.

Price of robustness: Z_rob(Gamma, radius) / Z_nom, both exactly verified
revenues of v1.  Out-of-sample violation rate: fraction of DRAWS i.i.d.
relative residuals h_i ~ U[-radius, radius] (one per product, shared by its
menu) under which the realized margin sum_i omega_i (x_i - kappa q_i) g_i(h_i)
is negative, with g_i(h) = ghat_i (1 + h) for h <= 0 and ghat_i / (1 - h) for
h > 0 (the inverse residual map of the manuscript).  The robust and nominal
decisions are evaluated on the same draws.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import protocol as P

DRAWS = 10_000


def _surplus_terms(base, ids):
    f = float.fromhex
    k = f(base["kappa"])
    u, g = [], []
    for i, j in enumerate(ids):
        x = f(base["menus"][i][j])
        u.append(f(base["exposures"][i]) * (x - k * f(base["costs"][i])))
        g.append(f(base["demands"][i][j]))
    return np.asarray(u), np.asarray(g)


def violation_rate(u, g, h):
    real = np.where(h <= 0.0, g * (1.0 + h), g / (1.0 - h))
    return float(np.mean((real * u).sum(axis=1) < 0.0))


def main(v1_dir, out_path, salt):
    v1 = Path(v1_dir)
    rows = []
    bases = [b for b in P.bases() if b["panel_code"] == 1]
    for bi, b in enumerate(bases):
        base = json.loads((v1 / "instances" / f"{b['base_id']}.json").read_text())
        n = base["n"]
        for ri, rad in enumerate(P.RADII):
            rng = np.random.default_rng(np.random.SeedSequence([salt, bi, ri]))
            h = rng.uniform(-rad, rad, size=(DRAWS, n))
            nom = json.loads((v1 / "raw" / f"{b['base_id']}-G0-R{rad:.2f}-DNC.json").read_text())["result"]
            un, gn = _surplus_terms(base, nom["original_ids"])
            z_nom = nom["exact"]["exact_revenue"]
            viol_nom = violation_rate(un, gn, h)
            for glab, gam in P.gamma_grid(n):
                rec = json.loads((v1 / "raw" / f"{b['base_id']}-G{glab}-R{rad:.2f}-DNC.json").read_text())["result"]
                ur, gr = _surplus_terms(base, rec["original_ids"])
                rows.append(dict(base_id=b["base_id"], family=b["family"], gamma_label=glab, gamma=gam, radius=rad,
                                 z_rob=rec["exact"]["exact_revenue"], z_nom=z_nom,
                                 ratio=rec["exact"]["exact_revenue"] / z_nom,
                                 violation_rob=violation_rate(ur, gr, h), violation_nom=viol_nom,
                                 changed_products=int(sum(a != c for a, c in zip(rec["original_ids"],
                                                                                 nom["original_ids"])))))
    Path(out_path).write_text(json.dumps(dict(draws=DRAWS, salt=salt, rows=rows), indent=0))
    return rows
