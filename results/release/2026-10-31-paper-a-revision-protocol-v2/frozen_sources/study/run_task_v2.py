"""Run one protocol-v2 task and write its JSON record (never overwrites)."""
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

from study.run_task import _clean


def _call(M, V, enc, meth, tl):
    if meth == "HR":
        return M.run_hullround(enc, sweep=False, dominance=False, time_limit=tl)
    if meth == "HR-sweep":
        return M.run_hullround(enc, sweep=True, dominance=False, time_limit=tl)
    if meth == "PIPE":
        return V.run_pipeline(enc, time_limit=tl)
    if meth == "HR+BB":
        return M.run_exact_layer(enc, time_limit=tl)
    if meth == "DNC":
        r = M.run_dnc(enc, time_limit=tl)
        # v1 run_dnc labels an infeasible root as TIME_LIMIT_NO_INCUMBENT; a genuine
        # time-out needs elapsed > time limit (no v1 run was infeasible).
        if r["status"] == "TIME_LIMIT_NO_INCUMBENT" and r["elapsed"] <= tl:
            r["status"] = "INFEASIBLE"
            r["status_relabelled"] = True
        return r
    if meth == "MILP":
        return M.run_milp(enc, time_limit=tl)
    if meth == "MILP-SCIP":
        return V.run_milp_scip(enc, time_limit=tl)
    raise ValueError(meth)


def warm_up(I, M, V, meth):
    base = I.generate_base(1, 0, "regular", 4, 3, 0)
    _call(M, V, I.Encoded(I.primitive_data(base, 0.1, 2)), meth, 10.0)


def run(task, base_path, out_path):
    from study import instances as I, methods as M, methods_v2 as V

    warm_up(I, M, V, task["method"])
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
        r = _call(M, V, enc, meth, tl)
        r["cpu_time"] = time.process_time() - c0
        if meth in ("HR", "HR-sweep", "PIPE"):
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
