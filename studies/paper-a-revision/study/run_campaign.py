"""Generate frozen base portfolios and run every planned task.

Usage (through the storage-guard runner):
  python -m study.run_campaign --evidence DIR --stage instances
  python -m study.run_campaign --evidence DIR --stage run --workers 3 [--panel S]

Each task runs in its own subprocess with a hard kill at time_limit + grace;
a killed or crashed task is recorded as KILLED/ERROR, never dropped.
Existing result files are never overwritten, so the run stage is resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from study import protocol as P

GRACE = 300.0


def stage_freeze(ev: Path):
    import hashlib
    import platform
    import shutil

    if (ev / "FROZEN_SOURCES.sha256").exists():
        sys.exit("already frozen")
    ev.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    src = ev / "frozen_sources"
    src.mkdir()
    files = sorted([root / "PROTOCOL.md", root / "env.sh"] + list((root / "study").glob("*.py")))
    import robust_mckp
    exact = Path(robust_mckp.__file__).resolve().parent  # the installed library
    files += sorted(exact.rglob("*.py"))
    lines = []
    for f in files:
        rel = f.relative_to(root) if root in f.parents else Path("robust_mckp") / f.relative_to(exact)
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, src / rel)
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}")
    (ev / "FROZEN_SOURCES.sha256").write_text("\n".join(lines) + "\n")
    from importlib import metadata
    frz = "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()))
    (ev / "environment.txt").write_text(f"python {sys.version}\nplatform {platform.platform()}\n"
                                        f"machine {platform.machine()}\n\n{frz}")
    (ev / "planned_tasks.json").write_text(json.dumps(P.tasks(), indent=0))
    print("frozen", len(lines), "files;", len(P.tasks()), "planned tasks")


def stage_instances(ev: Path):
    from study import instances as I

    d = ev / "instances"
    d.mkdir(parents=True, exist_ok=True)
    index = []
    for b in P.bases():
        path = d / f"{b['base_id']}.json"
        if path.exists():
            base = json.loads(path.read_text())
        else:
            base = I.generate_base(P.STUDY_SALT, b["panel_code"], b["family"], b["n"], b["m"], b["rep"])
            path.write_text(json.dumps(base, sort_keys=True))
        index.append(dict(base_id=b["base_id"], sha256=I.base_sha256(base), file=path.name))
        print("instance", b["base_id"], flush=True)
    (ev / "instances_index.json").write_text(json.dumps(index, indent=1))


def _one(task, ev: Path, py: str):
    out = ev / "raw" / f"{task['task_id']}.json"
    if out.exists():
        return task["task_id"], "exists"
    base = ev / "instances" / f"{task['base_id']}.json"
    cmd = [py, "-m", "study.run_task", "--task-json", json.dumps(task), "--base", str(base), "--out", str(out)]
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


def stage_run(ev: Path, workers: int, panel: str | None, order: str):
    (ev / "raw").mkdir(exist_ok=True)
    tasks = [t for t in P.tasks() if panel is None or t["panel"] == panel]
    if order == "small-first":
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
    ap.add_argument("--stage", choices=("freeze", "instances", "validate", "run"), required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--order", default="small-first")
    a = ap.parse_args()
    ev = Path(a.evidence)
    if a.stage == "freeze":
        stage_freeze(ev)
    elif a.stage == "instances":
        stage_instances(ev)
    elif a.stage == "validate":
        from study import validate
        validate.main(str(ev / "validation.json"))
    else:
        stage_run(ev, a.workers, a.panel, a.order)
