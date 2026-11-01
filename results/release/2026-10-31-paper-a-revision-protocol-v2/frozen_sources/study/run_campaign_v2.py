"""Protocol v2 campaign: freeze, Panel H instances, robustness evaluation, runs.

Usage (through the storage-guard runner):
  python -m study.run_campaign_v2 --evidence DIR --stage freeze
  python -m study.run_campaign_v2 --evidence DIR --stage instances
  python -m study.run_campaign_v2 --evidence DIR --stage robustness
  python -m study.run_campaign_v2 --evidence DIR --stage run --workers 3 [--panel H]

v1 instances are read from the v1 evidence directory and checked against its
instances_index.json; nothing in the v1 directory is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from study import protocol_v2 as P

GRACE = 300.0
V1 = Path(P.V1_EVIDENCE)


def stage_freeze(ev: Path):
    import platform
    import shutil

    if (ev / "FROZEN_SOURCES.sha256").exists():
        sys.exit("already frozen")
    ev.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    src = ev / "frozen_sources"
    src.mkdir()
    files = sorted([root / "PROTOCOL.md", root / "PROTOCOL_V2.md", root / "env.sh"] +
                   list((root / "study").glob("*.py")))
    exact = Path("/Users/ericshao/.local/share/robust-mckp/worktrees/paper-a-exact-primitive-75d47042/src/robust_mckp")
    files += sorted(exact.glob("*.py"))
    lines = []
    for f in files:
        rel = f.relative_to(root) if root in f.parents else Path("robust_mckp_75d4704") / f.name
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, src / rel)
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}")
    (ev / "FROZEN_SOURCES.sha256").write_text("\n".join(lines) + "\n")
    frz = subprocess.run(["/Users/ericshao/.local/bin/uv", "pip", "freeze", "--python", sys.executable],
                         capture_output=True, text=True).stdout
    (ev / "environment.txt").write_text(f"python {sys.version}\nplatform {platform.platform()}\n"
                                        f"machine {platform.machine()}\n\n{frz}")
    (ev / "planned_tasks.json").write_text(json.dumps(P.tasks(), indent=0))
    print("frozen", len(lines), "files;", len(P.tasks()), "planned tasks")


def _v1_index():
    return {r["base_id"]: r["sha256"] for r in json.loads((V1 / "instances_index.json").read_text())}


def stage_instances(ev: Path):
    from study import instances as I
    from study import panel_h as PH

    d = ev / "instances"
    d.mkdir(parents=True, exist_ok=True)
    index = []
    for b in P.bases_h():
        path = d / f"{b['base_id']}.json"
        if path.exists():
            base = json.loads(path.read_text())
        else:
            base = PH.generate_base_h(P.STUDY_SALT_V2, b["n"], b["m"], b["rep"])
            path.write_text(json.dumps(base, sort_keys=True))
        index.append(dict(base_id=b["base_id"], sha256=I.base_sha256(base), file=path.name))
        print("instance", b["base_id"], flush=True)
    # hash check of the v1 instances used by protocol v2
    v1 = _v1_index()
    bad = []
    for bid in sorted({t["base_id"] for t in P.tasks() if t["instance_source"] == "v1"}):
        base = json.loads((V1 / "instances" / f"{bid}.json").read_text())
        if I.base_sha256(base) != v1[bid]:
            bad.append(bid)
    (ev / "instances_index.json").write_text(json.dumps(dict(panel_h=index, v1_checked=len(v1), v1_mismatch=bad),
                                                        indent=1))
    print("v1 instances checked:", len(v1), "mismatches:", bad)
    if bad:
        sys.exit("v1 instance hash mismatch")


def stage_robustness(ev: Path):
    from study import robustness as R

    out = ev / "robustness.json"
    if out.exists():
        sys.exit("robustness.json exists")
    rows = R.main(str(V1), str(out), P.STUDY_SALT_V2)
    print("robustness rows:", len(rows))


def _base_path(task, ev: Path):
    if task["instance_source"] == "v1":
        return V1 / "instances" / f"{task['base_id']}.json"
    return ev / "instances" / f"{task['base_id']}.json"


def _one(task, ev: Path, py: str):
    out = ev / "raw" / f"{task['task_id']}.json"
    if out.exists():
        return task["task_id"], "exists"
    cmd = [py, "-m", "study.run_task_v2", "--task-json", json.dumps(task), "--base", str(_base_path(task, ev)),
           "--out", str(out)]
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True,
                            timeout=task["time_limit"] + GRACE)
        if not out.exists():
            rec = dict(task=task, ok=False, error=f"no output; returncode {cp.returncode}",
                       stderr=cp.stderr[-4000:], started=t0, finished=time.time())
            out.write_text(json.dumps(rec, sort_keys=True))
            return task["task_id"], "error"
        return task["task_id"], "done"
    except subprocess.TimeoutExpired:
        rec = dict(task=task, ok=False, killed=True, error=f"killed after {task['time_limit'] + GRACE}s",
                   started=t0, finished=time.time())
        out.write_text(json.dumps(rec, sort_keys=True))
        return task["task_id"], "killed"


def stage_run(ev: Path, workers: int, panel: str | None):
    (ev / "raw").mkdir(exist_ok=True)
    tasks = [t for t in P.tasks() if panel is None or t["panel"] == panel]
    tasks.sort(key=lambda t: (t["n"] * t["m"], t["task_id"]))
    py = sys.executable
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tid, st in ex.map(lambda t: _one(t, ev, py), tasks):
            done += 1
            print(f"[{done}/{len(tasks)}] {tid} {st}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--stage", choices=("freeze", "instances", "robustness", "run"), required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--panel", default=None)
    a = ap.parse_args()
    ev = Path(a.evidence)
    {"freeze": lambda: stage_freeze(ev), "instances": lambda: stage_instances(ev),
     "robustness": lambda: stage_robustness(ev),
     "run": lambda: stage_run(ev, a.workers, a.panel)}[a.stage]()
