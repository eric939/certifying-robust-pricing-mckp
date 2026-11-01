# Evidence: Paper A TOP revision computational study, protocol v1

Directory: `/Volumes/LaCie/Archives/Codex/Projects/robust-mckp/Artifacts/paper-a-top-revision-study/2026-10-31-protocol-v1`

| File | Content |
|---|---|
| `FROZEN_SOURCES.sha256`, `frozen_sources/` | Hashes and copies of the protocol, study code and the exact primitive adapter (local commit 75d4704), frozen before instance generation |
| `environment.txt` | Python, platform and package versions |
| `planned_tasks.json` | All 1,764 planned (run, method) tasks |
| `instances/`, `instances_index.json` | Binary64 primitive inputs (hex) and SHA-256 of all 66 study base portfolios |
| `validation.json` | Panel V (180 runs) with every check |
| `raw/<task_id>.json` | One record per planned task: status, bounds, times, load averages, exact rational check, original menu IDs |
| `campaign.log`, `campaign_launch_attempt0.log` | Scheduler log; attempt 0 failed before running anything (shell quoting) |
| `all_runs.csv` | One row per planned task (written by `study/summarize.py`) |
| `summary_macros.json` | Values of the manuscript macros |
| `reporting/` | Reporting addendum (`report.py`), VALIDATION_NOTE.md, generated LaTeX and figure as delivered to the manuscript |
| `MANIFEST.sha256` | SHA-256 of every file above |

Reproduce: `source env.sh` from `paper-a/top-revision-study`, then
`python -m study.run_campaign --evidence DIR --stage {freeze,instances,validate,run}`
and `python -m study.report DIR GENERATED FIGURES`.

The reporting addendum `study/report.py` was written after the freeze. It
changes no run; it splits tables, typesets macros, redraws the figure and writes
the supplementary per-configuration tables and the manifest. Its hash is
recorded in `MANIFEST.sha256` (under `reporting/`).

Reporting revision (31 Oct 2026, after the referee simulation): `report.py`
additionally writes `study_table_absolute.tex` (absolute objective values and
bounds per panel, size and method), draws Figure panel (c) as box plots
(interquartile range, median, minimum and maximum) with per-radius medians, and
titles the supplement "Online Resource 1". No run was repeated or changed; the
manifest was regenerated.
