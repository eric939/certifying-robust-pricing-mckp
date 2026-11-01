"""Pilot timings on pilot seeds (salt 20261030); not part of the evidence."""
import sys, time
import numpy as np
from study import instances as I, methods as Mth
PILOT = 20261030
cfgs = [(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))]
gam = sys.argv[4]; rad = float(sys.argv[5]); meths = sys.argv[6].split(','); tl = float(sys.argv[7])
for fam, n, m in cfgs:
    t0 = time.time(); base = I.generate_base(PILOT, 0, fam, n, m, 0)
    g = {'0': 0, 'sqrt': int(np.sqrt(n)), '0.1n': n // 10, '0.5n': n // 2, 'n': n}[gam]
    enc = I.Encoded(I.primitive_data(base, rad, g)); te = time.time() - t0
    ms = enc.menu_statistics()
    print(f'{fam} n={n} m={m} G={g} r={rad} M={enc.M} K={len(enc.thresholds())} gen+enc={te:.1f}s ratio={ms["menu_ratio"]:.2e}', flush=True)
    for meth in meths:
        t0 = time.time()
        if meth == 'HR': r = Mth.run_hullround(enc, sweep=False, dominance=False, time_limit=tl)
        if meth == 'HS': r = Mth.run_hullround(enc, sweep=True, dominance=False, time_limit=tl)
        if meth == 'HD': r = Mth.run_hullround(enc, sweep=True, dominance=True, time_limit=tl)
        if meth == 'BB': r = Mth.run_exact_layer(enc, time_limit=tl)
        if meth == 'DN': r = Mth.run_dnc(enc, time_limit=tl)
        if meth == 'MI': r = Mth.run_milp(enc, time_limit=tl)
        lb = r.get('objective', r.get('H')); ub = r.get('bound', r.get('U'))
        ex = enc.exact_check(r['sel']) if r.get('sel') is not None else None
        extra = {k: r[k] for k in ('thetas_evaluated','hull_builds','hull_build_bound','thetas_searched','nodes','interval_evaluations','H_rd') if k in r}
        print(f'  {meth}: {r["status"]} t={time.time()-t0:.1f}s lb={lb} ub={ub} gap={Mth._gap(lb,ub)} exact={None if ex is None else ex["exact_feasible"]} {extra}', flush=True)
