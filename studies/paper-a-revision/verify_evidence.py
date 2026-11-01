"""Verify the evidence of the revised paper's computational study (both protocols).

Run from this folder after ``source env.sh``::

    python verify_evidence.py                 # integrity + regenerated reports (minutes)
    python verify_evidence.py --instances     # + regenerate all base portfolios from their seeds
    python verify_evidence.py --rerun 60      # + re-run 60 sampled runs and compare results
    python verify_evidence.py --instances --write-instances   # also store v1 instances for a full rerun

Checks, per protocol:

1. every file listed in MANIFEST.sha256 is present with its hash, except the
   files the public tree omits because they are regenerated: the protocol-v1
   base portfolios (``instances/``, 139 MB) and the generated LaTeX tables and
   figure (``reporting/*.tex``, ``reporting/*.pdf``);
2. FROZEN_SOURCES.sha256 matches the frozen copies of the protocol and code;
3. planned_tasks.json equals the task list of ``study.protocol`` /
   ``study.protocol_v2``, and every planned task has exactly one raw record;
4. all derived files (per-run CSVs, summary JSON, every LaTeX table and the
   figure) are rebuilt from the raw records alone in a scratch copy and must
   equal the manifest byte for byte;
5. with ``--instances``: every base portfolio is regenerated from its seed and
   must match instances_index.json and the manifest;
6. with ``--rerun K``: K runs (small instances, deterministic sample) are
   repeated with the library; status, objective, bound and original menu IDs
   must equal the record exactly for the combinatorial methods, and up to the
   1e-6 relative tolerance for the MILP solvers.  Runs that ended at a time
   limit are not sampled, since their outcome depends on machine speed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASES = HERE.parents[1] / "results" / "release"
V1 = RELEASES / "2026-10-31-paper-a-revision-protocol-v1"
V2 = RELEASES / "2026-10-31-paper-a-revision-protocol-v2"
DETERMINISTIC = {"HR", "HR-sweep", "HR-dom", "HR+BB", "DNC", "PIPE"}


class Failure(Exception):
    pass


def require(cond, msg):
    if not cond:
        raise Failure(msg)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest(ev: Path) -> dict:
    out = {}
    for line in (ev / "MANIFEST.sha256").read_text().splitlines():
        h, rel = line.split("  ", 1)
        out[rel] = h
    return out


def regenerated(rel: str) -> bool:
    return rel.startswith("instances/") or (rel.startswith("reporting/") and rel.endswith((".tex", ".pdf")))


def check_integrity(ev: Path) -> int:
    man = manifest(ev)
    present = {str(p.relative_to(ev)) for p in ev.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"}
    extra = sorted(present - set(man))
    require(not extra, f"{ev.name}: files not in the manifest: {extra[:5]}")
    n = 0
    for rel, h in man.items():
        p = ev / rel
        if not p.exists():
            require(regenerated(rel), f"{ev.name}: missing {rel}")
            continue
        require(sha(p) == h, f"{ev.name}: hash mismatch {rel}")
        n += 1
    for line in (ev / "FROZEN_SOURCES.sha256").read_text().splitlines():
        h, rel = line.split("  ", 1)
        require(sha(ev / "frozen_sources" / rel) == h, f"{ev.name}: frozen source mismatch {rel}")
    return n


def check_design(ev: Path, tasks: list) -> None:
    planned = json.loads((ev / "planned_tasks.json").read_text())
    require(planned == tasks, f"{ev.name}: planned_tasks.json differs from the protocol module")
    raw = {p.stem for p in (ev / "raw").glob("*.json")}
    ids = [t["task_id"] for t in tasks]
    require(len(set(ids)) == len(ids), f"{ev.name}: duplicate task IDs")
    require(raw == set(ids), f"{ev.name}: raw records and planned tasks differ "
                             f"({len(raw - set(ids))} extra, {len(set(ids) - raw)} missing)")


def check_reports(tmp: Path) -> None:
    """Rebuild every derived file from the raw records and compare to the manifests."""
    derived = ("all_runs.csv", "all_runs_v2.csv", "summary_macros.json", "summary_v2.json")
    copies = {}
    for name, ev in (("v1", V1), ("v2", V2)):
        dst = tmp / name
        shutil.copytree(ev, dst, ignore=shutil.ignore_patterns("reporting"))
        for f in derived:
            (dst / f).unlink(missing_ok=True)
        copies[name] = dst
    gen, fig = tmp / "generated", tmp / "figures"
    gen.mkdir()
    fig.mkdir()
    env = dict(os.environ, PAPER_A_V1_EVIDENCE=str(copies["v1"]),
               PYTHONPATH=os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")]))
    for mod, ev in (("study.report", copies["v1"]), ("study.report_v2", copies["v2"])):
        r = subprocess.run([sys.executable, "-m", mod, str(ev), str(gen), str(fig)], cwd=HERE, env=env,
                           capture_output=True, text=True)
        require(r.returncode == 0, f"{mod} failed:\n{r.stderr[-2000:]}")
    import matplotlib

    recorded = (V2 / "environment.txt").read_text()
    same_mpl = f"matplotlib=={matplotlib.__version__}\n" in recorded
    for name, ev in (("v1", V1), ("v2", V2)):
        bad = []
        for rel, h in manifest(ev).items():
            if rel.startswith("instances/"):
                continue
            if rel.endswith(".pdf") and not same_mpl:
                continue  # the PDF bytes depend on the matplotlib version
            p = copies[name] / rel
            if not p.exists() or sha(p) != h:
                bad.append(rel)
        require(not bad, f"{ev.name}: regenerated files differ from the manifest: {bad[:8]}")
    return same_mpl


def check_instances(write: bool) -> int:
    from study import instances as I, panel_h as PH, protocol as P1, protocol_v2 as P2

    n = 0
    for ev, bases, gen in ((V1, P1.bases(), lambda b: I.generate_base(P1.STUDY_SALT, b["panel_code"], b["family"],
                                                                      b["n"], b["m"], b["rep"])),
                           (V2, P2.bases_h(), lambda b: PH.generate_base_h(P2.STUDY_SALT_V2, b["n"], b["m"],
                                                                           b["rep"]))):
        rows = json.loads((ev / "instances_index.json").read_text())
        rows = rows["panel_h"] if isinstance(rows, dict) else rows  # v2 also records its v1 hash check
        index = {r["base_id"]: r for r in rows}
        man = manifest(ev)
        for b in bases:
            base = gen(b)
            row = index[b["base_id"]]
            require(I.base_sha256(base) == row["sha256"], f"{ev.name}: instance {b['base_id']} differs")
            text = json.dumps(base, sort_keys=True)
            require(hashlib.sha256(text.encode()).hexdigest() == man["instances/" + row["file"]],
                    f"{ev.name}: instance file {row['file']} differs from the manifest")
            if write and ev == V1:
                (ev / "instances").mkdir(exist_ok=True)
                (ev / "instances" / row["file"]).write_text(text)
            n += 1
    return n


def _close(a, b, rel):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= rel * max(1.0, abs(b))


def check_rerun(k: int) -> tuple:
    import numpy as np
    from study import instances as I, panel_h as PH, protocol as P1, protocol_v2 as P2
    from study import methods as M, methods_v2 as V, run_task as T1, run_task_v2 as T2

    pool = []
    for ev in (V1, V2):
        for p in sorted((ev / "raw").glob("*.json")):
            rec = json.loads(p.read_text())
            t = rec["task"]
            if not rec.get("ok") or t["n"] * t["m"] > 3200:
                continue
            if not str(rec["result"].get("status", "")).startswith(("OPTIMAL", "CERTIFIED", "INFEASIBLE")):
                continue
            pool.append((ev, rec))
    sample = random.Random(20261101).sample(pool, min(k, len(pool)))
    cache = {}
    exact_ok = milp_ok = 0
    for ev, rec in sample:
        t = rec["task"]
        key = (t.get("instance_source", "v1"), t["base_id"])
        if key not in cache:
            if t["panel"] == "H":
                cache[key] = PH.generate_base_h(P2.STUDY_SALT_V2, t["n"], t["m"], t["rep"])
            else:
                cache[key] = I.generate_base(P1.STUDY_SALT, t["panel_code"], t["family"], t["n"], t["m"], t["rep"])
        enc = I.Encoded(I.primitive_data(cache[key], t["radius"], t["gamma"]))
        if ev == V1:
            r = T1._call(M, enc, t["method"], t["time_limit"])
        else:
            r = T2._call(M, V, enc, t["method"], t["time_limit"])
        if "H" in r:
            r["objective"], r["bound"] = r["H"], r["U"]
        ids = enc.original_ids(r["sel"]) if r.get("sel") is not None else None
        old = rec["result"]
        status = r["status"]
        where = f"{ev.name}/{t['task_id']}"
        require(status == old["status"], f"rerun {where}: status {status} != {old['status']}")
        if t["method"] in DETERMINISTIC:
            require(r["objective"] == old["objective"] and r["bound"] == old["bound"],
                    f"rerun {where}: ({r['objective']}, {r['bound']}) != ({old['objective']}, {old['bound']})")
            require(ids == old["original_ids"], f"rerun {where}: decision differs")
            exact_ok += 1
        else:
            require(_close(r["objective"], old["objective"], 1e-6), f"rerun {where}: MILP objective differs")
            milp_ok += 1
    return exact_ok, milp_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", action="store_true", help="regenerate every base portfolio and check it")
    ap.add_argument("--write-instances", action="store_true", help="store the regenerated v1 portfolios")
    ap.add_argument("--rerun", type=int, default=0, metavar="K", help="re-run K sampled runs")
    ap.add_argument("--skip-reports", action="store_true", help="skip rebuilding the derived files")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    from study import protocol as P1, protocol_v2 as P2

    try:
        n1, n2 = check_integrity(V1), check_integrity(V2)
        print(f"integrity: {n1} + {n2} files match their manifests; frozen sources match")
        check_design(V1, P1.tasks())
        check_design(V2, P2.tasks())
        print(f"design: {len(P1.tasks())} + {len(P2.tasks())} planned runs, one raw record each")
        if not a.skip_reports:
            with tempfile.TemporaryDirectory() as d:
                same_mpl = check_reports(Path(d))
            print("reports: every CSV, summary and table" + (" and the figure" if same_mpl else "") +
                  " rebuilt from the raw records, byte for byte" +
                  ("" if same_mpl else " (figure not compared: install requirements-study.txt for its matplotlib)"))
        if a.instances or a.write_instances:
            print(f"instances: {check_instances(a.write_instances)} base portfolios regenerated from their seeds")
        if a.rerun:
            e, m = check_rerun(a.rerun)
            print(f"rerun: {e} combinatorial runs identical, {m} MILP runs within 1e-6")
    except Failure as exc:
        print("PAPER A REVISION EVIDENCE: FAIL:", exc)
        return 1
    print("PAPER A REVISION EVIDENCE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
