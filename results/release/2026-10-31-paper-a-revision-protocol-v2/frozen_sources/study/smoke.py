"""Development smoke test: all methods vs brute force on tiny instances."""
import sys, time
import numpy as np
t0 = time.time()
from study import instances as I, methods as Mth
print('import', round(time.time() - t0, 1), flush=True)
bad = 0
cnt = 0
for fam in ['regular', 'charm', 'noisy']:
    for rep in range(3):
        base = I.generate_base(99, 9, fam, 7, 4, rep)
        for g in [0, 1, 3, 7]:
            for rad in [0.1, 0.3]:
                enc = I.Encoded(I.primitive_data(base, rad, g))
                bf = Mth.run_bruteforce(enc)
                hr = Mth.run_hullround(enc, sweep=False, dominance=False, time_limit=60)
                hs = Mth.run_hullround(enc, sweep=True, dominance=False, time_limit=60)
                hd = Mth.run_hullround(enc, sweep=True, dominance=True, time_limit=60)
                ex = Mth.run_exact_layer(enc, time_limit=60)
                dn = Mth.run_dnc(enc, time_limit=60)
                mi = Mth.run_milp(enc, time_limit=60)
                opt = bf['objective']
                cnt += 1
                def close(a, b):
                    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-6 * max(1, abs(b)))
                ok = close(ex['objective'], opt) and close(dn['objective'], opt) and close(mi['objective'], opt)
                ok = ok and (opt is None or (hr['H'] <= opt + 1e-9 and hr['U'] >= opt - 1e-6 * abs(opt)))
                ok = ok and close(hr['H'], hs['H']) and close(hr['U'], hs['U']) and close(hd['H'], hs['H'])
                if not ok:
                    bad += 1
                    print(fam, rep, g, rad, 'BF', opt, 'HR', hr['H'], hr['U'], 'HS', hs['H'], hs['U'], 'HD', hd['H'], hd['U'],
                          'EX', ex['status'], ex['objective'], 'DN', dn['status'], dn['objective'], 'MI', mi['status'], mi['objective'], flush=True)
print('cases', cnt, 'bad', bad, 'time', round(time.time() - t0, 1), flush=True)
