# Reproducibility Guide

This repository tracks the solver, experiment code, immutable arXiv-v1
provenance, and the editable arXiv-v2 manuscript source under `paper/current`.
Generated result files, manuscript builds, and submission packages are written
outside the source tree. Submission-package details live in `SUBMISSION.md`;
scientific guardrails and latest validation status live in
`REVISION_HISTORY.md`.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[experiments,validation,dev]"
```

Optional baselines use PySCIPOpt/SCIP, HiGHS/SciPy, Gurobi, and CPLEX when
available. The core solver and smoke checks do not require commercial solvers.

## Bounded Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_parametric_theta_sweep.py tests/test_segment_local_budgets.py -q
.venv/bin/python scripts/run_clean_repro_check.py --quick
```

The bounded check verifies imports and tests, py-compiles the experiment
scripts, runs a small publication benchmark, creates synthetic Path C
calibration data, and runs a small semi-synthetic pricing experiment. Outputs
are written under `results/clean_repro_check/`.

For a clean-copy check:

```bash
.venv/bin/python scripts/run_true_clean_room_check.py --work-dir /tmp/robust_mckp_clean_check --quick
```

This copies the repository to a temporary directory, installs it in a fresh
virtual environment, and runs the same bounded checks there.

## Experiment Entry Points

## Parametric θ-sweep benchmark

The sweep is exact because it evaluates the same full original breakpoint set
as independent enumeration and validates or rebuilds hulls when necessary.

Smoke run:

```bash
.venv/bin/python scripts/run_parametric_sweep_ablation.py --smoke --output-dir results/parametric_sweep_ablation_smoke --validate
.venv/bin/python scripts/summarize_parametric_sweep_ablation.py --input-dir results/parametric_sweep_ablation_smoke --tables-dir results/parametric_sweep_summaries/tables --label smoke
.venv/bin/python scripts/run_parametric_sweep_benchmarks.py --smoke --output-dir results/parametric_sweep_smoke --validate-sweep-sampled
.venv/bin/python scripts/summarize_parametric_sweep_benchmarks.py --input-dir results/parametric_sweep_smoke --tables-dir results/parametric_sweep_summaries/tables --label smoke
.venv/bin/python scripts/run_segment_local_budget_experiments.py --smoke --output-dir results/segment_local_budget_smoke
.venv/bin/python scripts/summarize_segment_local_budget_experiments.py --input-dir results/segment_local_budget_smoke --tables-dir results/parametric_sweep_summaries/tables --label smoke
.venv/bin/python scripts/build_parametric_sweep_claim_audit.py
```

Normal-run template:

```bash
.venv/bin/python scripts/run_parametric_sweep_ablation.py \
  --paper-lite \
  --families many_theta,tight_capacity,non_tight_control \
  --output-dir results/parametric_sweep_ablation_lite \
  --validate-sampled \
  --resume
.venv/bin/python scripts/summarize_parametric_sweep_ablation.py \
  --input-dir results/parametric_sweep_ablation_lite \
  --tables-dir results/parametric_sweep_summaries/tables \
  --label lite
.venv/bin/python scripts/run_parametric_sweep_benchmarks.py \
  --paper-lite \
  --families many_theta,tight_capacity,non_tight_control \
  --methods hullround,exact_enum_current,exact_sweep_new,scipy_highs,scip,gurobi,cplex \
  --output-dir results/parametric_sweep_lite \
  --time-limit 45 \
  --node-limit 300000 \
  --validate-sweep-sampled \
  --resume
.venv/bin/python scripts/summarize_parametric_sweep_benchmarks.py \
  --input-dir results/parametric_sweep_lite \
  --tables-dir results/parametric_sweep_summaries/tables \
  --label lite
```

Outputs are CSV files under the selected `results/` directory and generated
LaTeX tables under `results/parametric_sweep_summaries/tables/`. Optional
figures, if added, should be written under
`results/parametric_sweep_summaries/figures/`.

The ablation runner compares independent fixed-θ rebuilding,
incremental-sweep rebuilding, and incremental-sweep safe reuse without full
B&B. It supports the narrower construction-cost claim only. Solver-level
speedups require the full benchmark summary and matched certified rows.

Segment-local Gamma budgets are smoke-validated with brute-force parity and a
product-size guard. Larger segment-local products can grow quickly; oversized
rows should be reported as guarded or partial rather than approximated.

SCIP, HiGHS/SciPy, Gurobi, and CPLEX rows are included only when the relevant
package and license are available. Unavailable backends are recorded as
`not_available` instead of failing smoke runs.

Synthetic robust-MCKP smoke benchmark:

```bash
.venv/bin/python scripts/run_publication_benchmarks.py --smoke --output-dir results/publication_benchmarks_smoke
```

Publishable synthetic/retail experiment bundle:

```bash
.venv/bin/python scripts/run_publishable_experiments.py --smoke
.venv/bin/python scripts/run_solver_benchmarks.py --smoke
.venv/bin/python scripts/plot_publishable_results.py
```

The HullRound gap CSVs include the additive one-item certificate
`delta_v_max_theta`, the realized round-down loss `l_rd`, the certificate ratio
`l_rd_over_delta`, and the scale-normalized diagnostic
`delta_v_max_over_lp_ub`.

Semi-synthetic pricing application:

```bash
.venv/bin/python scripts/run_pathC_data_calibration.py --source synthetic_only --output-dir results/pathC/calibration
.venv/bin/python scripts/run_pathC_semisynthetic_application.py --calibration-dir results/pathC/calibration --output-dir results/pathC/semisynthetic_application_smoke --seeds 1 --n 60 --m 8 --stress-scenarios 200 --gamma-grid 0,sqrt,n --run-exact-small-subset
```

Diagnostics:

```bash
.venv/bin/python scripts/run_branching_strategy_diagnostics.py
.venv/bin/python scripts/run_tight_capacity_diagnostics.py
```

Most scripts accept `--output-dir` or related path options. Use those options to
keep new outputs separate from previous local runs.
