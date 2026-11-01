# Evidence: Paper A TOP revision computational study, protocol v2

Directory: `/Volumes/LaCie/Archives/Codex/Projects/robust-mckp/Artifacts/paper-a-top-revision-study/2026-10-31-protocol-v2`
(v1 evidence next to it is read only, never written.)

| File | Content |
|---|---|
| `FROZEN_SOURCES.sha256`, `frozen_sources/` | Protocol v2 (`PROTOCOL_V2.md`), study code and exact adapter, hashed before any v2 instance |
| `environment.txt`, `planned_tasks.json` | Environment (incl. PySCIPOpt 6.3.0 / SCIP 10.0) and the 1,260 planned tasks |
| `instances/`, `instances_index.json` | 24 Panel H base portfolios; hash check of the 66 v1 instances (0 mismatches) |
| `robustness.json` | Price of robustness / out-of-sample violations of the v1 Panel S DnC decisions |
| `raw/<task_id>.json` | One record per planned task |
| `campaign.log` | Scheduler log (1,260/1,260, 0 killed, 0 errors) |
| `all_runs_v2.csv`, `summary_v2.json` | Rows of v1 and v2 used for reporting; macros and comparisons |
| `reporting/` | `report_v2.py` (written after the v2 freeze; changes no run), generated tables, figure |
| `MANIFEST.sha256` | SHA-256 of every file above |

Reproduce: `launch_v2.sh --stage {freeze,instances,robustness,run}` via the storage-guard runner, then
`python -m study.report <v1> GENERATED FIGURES` followed by `python -m study.report_v2 <v2> GENERATED FIGURES`.
Failures: 5 SCIP decisions reported optimal but infeasible in exact arithmetic (residual -0.03 to -1.0).
