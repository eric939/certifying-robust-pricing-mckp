# Computational study of the revised paper

Code of the study in Section 7 of the revised paper. Its evidence is in
`results/release/2026-10-31-paper-a-revision-protocol-v1` and `-v2`.

| File | Role |
|---|---|
| `PROTOCOL.md`, `study/protocol.py` | protocol v1: Panels S and L, 1,764 runs, fixed before the instances were generated |
| `PROTOCOL_V2.md`, `study/protocol_v2.py` | protocol v2: interval records and SCIP on the v1 instances, Panel H, price of robustness; 1,260 runs |
| `study/instances.py`, `study/panel_h.py` | seeded generators of the base portfolios and the exact primitive encoding |
| `study/core.py`, `study/methods.py`, `study/methods_v2.py` | the kernels and solver drivers; they now re-export `robust_mckp.fast`, which is the same code |
| `study/run_campaign*.py`, `study/run_task*.py` | campaign drivers: one subprocess per run, hard kill at limit + 300 s, never overwrite |
| `study/robustness.py` | price-of-robustness evaluation (protocol v2) |
| `study/validate.py` | validation panel V: every method against brute force and the exact certificate |
| `study/summarize.py`, `study/report.py`, `study/report_v2.py` | per-run CSVs, the paper's tables and macros, Online Resource 1 and Figure 1 |
| `verify_evidence.py` | verifies both evidence releases (see `REPRODUCIBILITY.md`) |
| `VALIDATION_NOTE.md` | the one failing validation check and its cause (fixed in v4) |
| `EVIDENCE_README*.md` | the README of each evidence release (copied into it by the report generators) |
| `submitted_check/` | re-check of the submitted version's evidence (run against tag `v3.1.1`) |
| `requirements-study.txt` | the campaigns' exact package versions |
| `env.sh` | single-threaded kernels and this folder on `PYTHONPATH` |

The evidence READMEs and frozen sources record the paths of the author's
machine where the campaigns ran (`/Users/...`, `/Volumes/LaCie/...`); they are
kept as recorded, since the manifests cover them. The code here takes its
paths from arguments or from `PAPER_A_V1_EVIDENCE`.

The pilot scripts (`pilot.py`, `pilot_h.py`, `smoke.py`) are the checks run
before each protocol was fixed, with a separate seed; no reported number
comes from them.
