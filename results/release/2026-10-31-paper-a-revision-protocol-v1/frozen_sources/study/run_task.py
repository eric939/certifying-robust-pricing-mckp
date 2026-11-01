"""Run one planned task and write its JSON record (never overwrites)."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _call(M, enc, meth, tl):
    if meth == "HR":
        return M.run_hullround(enc, sweep=False, dominance=False, time_limit=tl)
    if meth == "HR-sweep":
        return M.run_hullround(enc, sweep=True, dominance=False, time_limit=tl)
    if meth == "HR-dom":
        return M.run_hullround(enc, sweep=True, dominance=True, time_limit=tl)
    if meth == "HR+BB":
        return M.run_exact_layer(enc, time_limit=tl)
    if meth == "DNC":
        return M.run_dnc(enc, time_limit=tl)
    if meth == "MILP":
        return M.run_milp(enc, time_limit=tl)
    raise ValueError(meth)


def warm_up(I, M, meth):
    """Untimed call on a tiny instance so JIT-cache loading is not timed."""
    base = I.generate_base(1, 0, "regular", 4, 3, 0)
    _call(M, I.Encoded(I.primitive_data(base, 0.1, 2)), meth, 10.0)


def run(task, base_path, out_path):
    from study import instances as I, methods as M

    warm_up(I, M, task["method"])
    rec = dict(task=task, host=platform.node(), python=sys.version.split()[0], pid=os.getpid(),
               started=time.time(), load_before=os.getloadavg())
    try:
        base = json.loads(Path(base_path).read_text())
        t0 = time.perf_counter()
        enc = I.Encoded(I.primitive_data(base, task["radius"], task["gamma"]))
        rec["encode_time"] = time.perf_counter() - t0
        rec["instance"] = dict(n=enc.n, M=enc.M, original_options=enc.n_original_options,
                               thresholds=int(np.unique(np.concatenate([[0.0], enc.t])).size),
                               **enc.menu_statistics())
        meth = task["method"]
        tl = task["time_limit"]
        c0 = time.process_time()
        r = _call(M, enc, meth, tl)
        r["cpu_time"] = time.process_time() - c0
        if meth.startswith("HR") and meth != "HR+BB":
            r["objective"] = r["H"]
            r["bound"] = r["U"]
            r["exact"] = enc.exact_check(r["sel"]) if r["sel"] is not None else None
            r["exact_rd"] = enc.exact_check(r["sel_rd"]) if r.get("sel_rd") is not None else None
            r["original_ids_rd"] = enc.original_ids(r["sel_rd"]) if r.get("sel_rd") is not None else None
        else:
            r["exact"] = enc.exact_check(r["sel"]) if r.get("sel") is not None else None
        r["original_ids"] = enc.original_ids(r["sel"]) if r.get("sel") is not None else None
        r.pop("sel", None)
        r.pop("sel_rd", None)
        rec["result"] = r
        rec["ok"] = True
    except Exception as exc:  # retained as an ERROR outcome
        rec["ok"] = False
        rec["error"] = repr(exc)
        rec["traceback"] = traceback.format_exc()
    rec["finished"] = time.time()
    rec["load_after"] = os.getloadavg()
    out = Path(out_path)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(_clean(rec), sort_keys=True))
    os.replace(tmp, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-json", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"refusing to overwrite {a.out}")
    run(json.loads(a.task_json), a.base, a.out)
