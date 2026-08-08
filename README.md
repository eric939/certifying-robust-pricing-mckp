# robust_pricing_mckp

Paper A repository for the manuscript
**“Certifying Γ-Robust Discrete Pricing via Full-Breakpoint
Multiple-Choice Knapsack Decomposition.”**  The package implements
finite-menu integer-budget Γ-robust MCKP routines, including HullRound,
exact θ-enumerated branch-and-bound, and an exact parametric θ-sweep with
incremental fixed-θ data and safe hull maintenance.

This is the independent continuation of
[arXiv:2603.18653](https://arxiv.org/abs/2603.18653). The active manuscript is
being prepared as version 2 of that arXiv record. It retains the original
paper's identity—robust finite-menu pricing, full-breakpoint decomposition,
fixed-threshold MCKP structure, rounding, and exact certification—and does not
include the simultaneous all-threshold envelope contribution of the separate
Paper B repository.

The related Paper B project,
[robust_mckp](https://github.com/eric939/robust_mckp), studies simultaneous
group-envelope bounds over the threshold family. It builds on and cites Paper A
but has a different central theorem, algorithm, evidence release, and arXiv
identifier. See `PAPER_LINEAGE.md` for the formal contribution boundary.

## Contents

- `README.md`: project overview and quick commands.
- `REPRODUCIBILITY.md`: environment, tests, smoke checks, and experiment entry points.
- `SUBMISSION.md`: public/blind manuscript and artifact-package checklist.
- `REVISION_HISTORY.md`: durable scientific guardrails and latest validation status.
- `PAPER_LINEAGE.md`: exact boundary and citation relationship with Paper B.
- `paper/current/`: active editable source for arXiv:2603.18653v2.
- `paper/archive/arxiv-2603.18653v1/`: immutable March 2026 PDF, source,
  and code archive.
- `manuscript/`: recovered complete 39-page public V3 paper and provenance.
- `src/robust_mckp/`: solver package, including HullRound and exact
  θ-enumerated branch-and-bound.
- `src/robust_mckp/parametric_sweep.py`: exact full-breakpoint parametric
  θ sweep with incremental `s^theta` updates and safe hull reuse/rebuilds.
- `src/robust_mckp/local_budget.py`: exact segment-local Gamma-budget extension
  for small θ-vector products.
- `src/robust_mckp/milp_baselines.py`: optional θ-decomposition MILP
  baselines when solver packages are installed/licensed.
- `experiments_nested/`: synthetic and nested robust-MCKP experiment drivers.
- `experiments_case_retail/`: semi-synthetic pricing experiment driver.
- `scripts/`: benchmark, plotting, summarization, and reproducibility scripts.
- `scripts/run_parametric_sweep_benchmarks.py`: many-θ, tight-capacity, and
  control benchmark driver for the exact sweep.
- `scripts/run_parametric_sweep_ablation.py`: data-construction and root-bound
  ablation for independent recomputation versus incremental sweep modes.
- `scripts/run_segment_local_budget_experiments.py`: guarded exact smoke
  experiments for segment-local Gamma budgets.
- `tests/`: public API, certificate, and exact-solver tests.

Generated outputs are written under `results/`, `paper_versions/`, `output/`,
`submission_ready/`, or other local artifact directories and are ignored by git.
The single canonical public PDF under `manuscript/` is deliberately retained.

## Main Algorithmic Contribution

- Finite full-breakpoint θ decomposition using
  `B = {0} union {|t_ij|}` from the original admissible options.
- Exact parametric sweep over the same full breakpoint set as enumeration.
- Incremental maintenance of `s^theta`, followed by exact recomputation of item
  baselines, fixed-θ costs, and capacity at every θ.
- Safe hull reuse only when the item cost-value point set is unchanged within
  tolerance; otherwise hulls are rebuilt exactly.
- Exact B&B still branches over original integer-safe non-dominated options,
  including below-upper-hull options.
- Every incumbent is final-checked by the original robust certificate.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[experiments,validation,dev]"
```

Minimal package-only install from GitHub after the repository is made public:

```bash
python3 -m pip install "git+https://github.com/eric939/robust_pricing_mckp.git"
```

## Quick Check

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_parametric_theta_sweep.py tests/test_segment_local_budgets.py -q
.venv/bin/python scripts/run_parametric_sweep_ablation.py --smoke --output-dir results/parametric_sweep_ablation_smoke --validate
.venv/bin/python scripts/run_parametric_sweep_benchmarks.py --smoke --output-dir results/parametric_sweep_smoke --validate-sweep-sampled
.venv/bin/python scripts/run_clean_repro_check.py --quick
```

The clean check runs tests, py-compiles the experiment scripts, and executes
small smoke experiments into `results/clean_repro_check/`.

## Reproducing Experiments

Smoke runs:

```bash
.venv/bin/python scripts/run_parametric_sweep_ablation.py --smoke --output-dir results/parametric_sweep_ablation_smoke --validate
.venv/bin/python scripts/summarize_parametric_sweep_ablation.py --input-dir results/parametric_sweep_ablation_smoke --tables-dir results/parametric_sweep_summaries/tables --label smoke
.venv/bin/python scripts/run_parametric_sweep_benchmarks.py --smoke --output-dir results/parametric_sweep_smoke --validate-sweep-sampled
.venv/bin/python scripts/summarize_parametric_sweep_benchmarks.py --input-dir results/parametric_sweep_smoke --tables-dir results/parametric_sweep_summaries/tables --label smoke
.venv/bin/python scripts/run_segment_local_budget_experiments.py --smoke --output-dir results/segment_local_budget_smoke
.venv/bin/python scripts/summarize_segment_local_budget_experiments.py --input-dir results/segment_local_budget_smoke --tables-dir results/parametric_sweep_summaries/tables --label smoke
.venv/bin/python scripts/run_publication_benchmarks.py --smoke --output-dir results/publication_benchmarks_smoke
.venv/bin/python scripts/run_publishable_experiments.py --smoke
.venv/bin/python scripts/run_solver_benchmarks.py --smoke
.venv/bin/python scripts/run_pathC_data_calibration.py --source synthetic_only --output-dir results/pathC/calibration
.venv/bin/python scripts/run_pathC_semisynthetic_application.py --calibration-dir results/pathC/calibration --output-dir results/pathC/semisynthetic_application_smoke --seeds 1 --n 60 --m 8 --stress-scenarios 200 --gamma-grid 0,sqrt,n --run-exact-small-subset
```

The parametric-sweep paper-lite templates are:

```bash
.venv/bin/python scripts/run_parametric_sweep_ablation.py --paper-lite --families many_theta,tight_capacity,non_tight_control --output-dir results/parametric_sweep_ablation_lite --validate-sampled --resume
.venv/bin/python scripts/run_parametric_sweep_benchmarks.py --paper-lite --families many_theta,tight_capacity,non_tight_control --methods hullround,exact_enum_current,exact_sweep_new,scipy_highs,scip,gurobi,cplex --time-limit 45 --node-limit 300000 --output-dir results/parametric_sweep_lite --validate-sweep-sampled --resume
```

Use generated CSV summaries for exactness, certification, and timing claims.
The repository is designed to report unavailable solver backends explicitly and
to avoid broad runtime-superiority claims unless a completed matched solver
campaign supports them.

Larger benchmark campaigns use the same scripts with the desired grid,
time-limit, and output-directory options. Optional exact-solver baselines use
SCIP, HiGHS/SciPy, Gurobi, and CPLEX only when installed and licensed;
commercial solvers are not required. Exact speed claims should be made only
from generated CSV summaries on matched rows.

## Citation

Please cite the software and accompanying manuscript. A machine-readable
citation file is included in `CITATION.cff`.

```bibtex
@misc{shao2026robustpricingmckp,
  title = {robust_pricing_mckp: Certifying Gamma-Robust Discrete Pricing via Full-Breakpoint MCKP Decomposition},
  author = {Shao, Eric},
  year = {2026},
  howpublished = {\url{https://github.com/eric939/robust_pricing_mckp}}
}
```

## License

MIT. See `LICENSE`.
