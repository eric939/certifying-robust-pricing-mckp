"""Pilot for Panel H on pilot seeds (salt 20261030); not part of the evidence."""
import sys, time
from study import instances as I, methods as M, methods_v2 as V, panel_h as PH
PILOT = 20261030
tl = float(sys.argv[1])
for spec in sys.argv[2:]:
    n, m, g, rad, rep, tau, fac, sig, e0, e1, dm = spec.split(":")
    n, m, rep, rad = int(n), int(m), int(rep), float(rad)
    g = {"sqrt": int(n ** 0.5), "0.5n": n // 2, "0.1n": n // 10}[g]
    base = PH.generate_base_h(PILOT, n, m, rep, float(tau), float(fac), float(sig), (float(e0), float(e1)), int(dm))
    enc = I.Encoded(I.primitive_data(base, rad, g))
    ms = enc.menu_statistics()
    print(f"spec={spec} n={n} m={m} G={g} r={rad} rep={rep} M={enc.M} ratio={ms['menu_ratio']:.3f}", flush=True)
    for name, fn in (("HR", lambda: M.run_hullround(enc, sweep=True, dominance=False, time_limit=tl)),
                     ("PIPE", lambda: V.run_pipeline(enc, time_limit=tl)),
                     ("BB", lambda: M.run_exact_layer(enc, time_limit=tl)),
                     ("DNC", lambda: M.run_dnc(enc, time_limit=tl)),
                     ("HiGHS", lambda: M.run_milp(enc, time_limit=tl))):
        t0 = time.time(); r = fn(); dt = time.time() - t0
        lb = r.get("objective", r.get("H")); ub = r.get("bound", r.get("U"))
        print(f"  {name:5s} {r['status']:10s} t={dt:7.2f} lb={lb} ub={ub} gap={M._gap(lb, ub)}", flush=True)
