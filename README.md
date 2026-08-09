# Certifying robust pricing MCKP — reproducibility code

This reviewer-facing repository contains the solver, tests, experiment drivers,
and released computational evidence for
[arXiv:2603.18653](https://arxiv.org/abs/2603.18653),
*A Certifying MCKP Framework for Gamma-Robust Discrete Pricing*.

The manuscript, submission packages, publication notes, and historical paper
sources are intentionally not part of this software repository.

## What is included

- `src/robust_mckp/`: the finite-menu robust-pricing/MCKP model, full
  binary64-distinct threshold construction, hull relaxation and rounding,
  exact branch-and-bound, parametric sweep, and direct robust certificate.
- `tests/`: correctness and adversarial regression tests.
- `experiments_nested/` and `experiments_case_retail/`: reproducible instance
  generators and experiment utilities.
- `scripts/`: experiment, mathematical-audit, benchmark, and summarization
  drivers.
- `results/release/2026-08-09-paper-a-final-r2/`: the compact canonical
  evidence release, including raw CSV rows, protocol, audit report, summary,
  generated evidence figure, and SHA-256 manifest.

## Install and verify

```bash
uv sync --extra experiments --extra validation --extra dev
make verify PYTHON=.venv/bin/python
```

This runs the complete test suite, verifies every released evidence byte, and
checks all scientific gates. A bounded fresh-code smoke reproduction is:

```bash
make reproduce-smoke PYTHON=.venv/bin/python
```

See `REPRODUCIBILITY.md` for the full audit and evidence-generation commands.
The software is MIT licensed.
