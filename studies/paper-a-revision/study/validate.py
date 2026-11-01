"""Validation panel V: compiled study implementations vs independent references.

For small instances (exhaustive enumeration feasible) every method is
compared with brute force; HullRound's H and U are compared with the
repository's reference implementation `robust_mckp.solve` and with the exact
rational global certificate `build_global_certificate`/`verify_global_certificate`
(local commit 75d4704), which recomputes U over the complete threshold family
in Fraction arithmetic.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from study import instances as I, methods as Mth
from study.protocol import STUDY_SALT, gamma_grid

from robust_mckp import solve_reference as ref_solve  # the v3 reference implementation
from robust_mckp.model import Option, PricingInstance  # noqa: E402
from robust_mckp.global_certificate import build_global_certificate, verify_global_certificate  # noqa: E402

PANEL_CODE = 9
import os
SALT = int(os.environ.get("STUDY_DEV_SALT", STUDY_SALT))  # dev override only
SIZES = ((6, 4), (8, 4), (10, 3))
RADII = (0.1, 0.3)
REPS = range(2)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


def main(out_path):
    rows = []
    t0 = time.time()
    for fam in I.FAMILIES:
        for (n, m) in SIZES:
            for rep in REPS:
                base = I.generate_base(SALT, PANEL_CODE, fam, n, m, rep)
                for glab, g in gamma_grid(n):
                    for rad in RADII:
                        enc = I.Encoded(I.primitive_data(base, rad, g))
                        row = dict(family=fam, n=n, m=m, rep=rep, gamma=g, gamma_label=glab, radius=rad,
                                   base_sha256=I.base_sha256(base))
                        bf = Mth.run_bruteforce(enc)
                        opt = bf["objective"]
                        row["opt"] = opt
                        hr = Mth.run_hullround(enc, sweep=False, dominance=False, time_limit=60)
                        hs = Mth.run_hullround(enc, sweep=True, dominance=False, time_limit=60)
                        ex = Mth.run_exact_layer(enc, time_limit=60)
                        dn = Mth.run_dnc(enc, time_limit=60)
                        mi = Mth.run_milp(enc, time_limit=60)
                        row.update(H=hr["H"], U=hr["U"], H_sweep=hs["H"], U_sweep=hs["U"],
                                   bb=ex["objective"], bb_status=ex["status"], dnc=dn["objective"],
                                   dnc_status=dn["status"], milp=mi["objective"], milp_status=mi["status"])
                        chk = {}
                        if opt is None:
                            chk["all_infeasible"] = (hr["status"] == "INFEASIBLE" and ex["status"] == "INFEASIBLE"
                                                     and dn["status"] == "INFEASIBLE" and mi["status"] == "INFEASIBLE")
                        else:
                            chk["bb"] = ex["status"] == "OPTIMAL" and rel(ex["objective"], opt) <= Mth.REL_TOL
                            chk["dnc"] = dn["status"] == "OPTIMAL" and rel(dn["objective"], opt) <= Mth.REL_TOL
                            chk["milp"] = mi["status"] == "OPTIMAL" and rel(mi["objective"], opt) <= Mth.REL_TOL
                            chk["interval"] = hr["H"] <= opt * (1 + 1e-12) and hr["U"] >= opt * (1 - 1e-12)
                            chk["sweep_equal"] = rel(hr["H"], hs["H"]) <= 1e-12 and rel(hr["U"], hs["U"]) <= 1e-12
                            chk["exact_H"] = bool(enc.exact_check(hr["sel"])["exact_feasible"])
                            chk["rd_bound"] = bool(hr["rd_bound_check"])
                            # repository reference heuristic on the same binary64 coefficients
                            items = [[Option(value=float(enc.v[k]), margin=float(enc.s[k]), uncertainty=float(enc.t[k]))
                                      for k in range(enc.off[i], enc.off[i + 1])] for i in range(enc.n)]
                            ref = ref_solve(PricingInstance(items, enc.gamma))
                            row["ref_H"] = ref.objective
                            chk["ref_H"] = rel(ref.objective, hr["H"]) <= 1e-9
                            # exact rational certificate of the returned decision
                            sel_local = [int(k - enc.off[i]) for i, k in enumerate(hr["sel"])]
                            cert = build_global_certificate(PricingInstance(enc.items, enc.gamma), sel_local)
                            ver = verify_global_certificate(PricingInstance(enc.items, enc.gamma), cert)
                            row["exact_U"] = float(ver.exact_u) if ver.exact_u is not None else None
                            chk["certificate_valid"] = bool(ver.valid)
                            chk["U_matches_exact"] = ver.exact_u is not None and rel(hr["U"], float(ver.exact_u)) <= 1e-9
                        row["checks"] = chk
                        row["pass"] = all(chk.values())
                        rows.append(row)
    summary = dict(cases=len(rows), passed=sum(r["pass"] for r in rows),
                   feasible=sum(r["opt"] is not None for r in rows),
                   failures=[r for r in rows if not r["pass"]], seconds=time.time() - t0)
    Path(out_path).write_text(json.dumps(dict(summary=summary, rows=rows), indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}), len(summary["failures"]), "failures")


if __name__ == "__main__":
    main(sys.argv[1])
