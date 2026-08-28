# Certifying robust pricing MCKP — v3 reproducibility code

This reviewer-facing repository contains the solver, exact certificate checker,
tests, experiment drivers, and released computational evidence for the current
v3 revision of
[arXiv:2603.18653](https://arxiv.org/abs/2603.18653),
*A Certifying MCKP Framework for Gamma-Robust Discrete Pricing*.
The paper-aligned code and evidence tag is
[`v3.0.1`](https://github.com/eric939/certifying-robust-pricing-mckp/tree/v3.0.1).

The manuscript, submission packages, publication notes, and historical paper
sources are intentionally not part of this software repository.

## What is included

- `src/robust_mckp/`: the finite-menu robust-pricing/MCKP model, full
  binary64-distinct threshold construction, hull relaxation and rounding,
  exact branch-and-bound, parametric sweep, direct robust-feasibility check,
  and exact full-family primal-dual certificate.
- `tests/`: correctness and adversarial regression tests.
- `experiments_nested/` and `experiments_case_retail/`: reproducible instance
  generators and experiment utilities.
- `scripts/`: experiment, mathematical-audit, benchmark, and summarization
  drivers.
- `results/release/2026-08-28-paper-a-v3/`: the canonical v3 evidence release:
  135 exact certificate intervals for 45 controlled instances, three exact
  pilot certificates, derived pilot records, protocols, provenance, source
  snapshot, exact binary64 input snapshots, and nested SHA-256 manifests.
- `results/release/2026-08-09-paper-a-final-r2/`: the frozen baseline release
  from which the v3 controlled instances were reconstructed.

## Install and verify

```bash
uv sync --extra experiments --extra validation --extra dev
make verify PYTHON=.venv/bin/python
```

This runs all 122 tests, checks both dated releases byte-for-byte, reconstructs
the 45 controlled instances, independently verifies all 138 v3 certificates,
and checks the public-tree hygiene boundary. A bounded fresh-code smoke
reproduction is:

```bash
make reproduce-smoke PYTHON=.venv/bin/python
```

See `REPRODUCIBILITY.md` for the full audit and evidence-generation commands.

The Dominick's pilot is post-hoc and exploratory. Only 7 of 79 future weeks
contain positive observed prices for the complete fixed roster, and the
availability/dependence gate fails. The repository therefore publishes the
derived record, exact optimization certificates, provenance, and source-file
hashes, but not the provider-controlled raw scanner files.

The software is MIT licensed. No manuscript source, manuscript PDF, submission
package, or restricted raw dataset is part of this repository.
