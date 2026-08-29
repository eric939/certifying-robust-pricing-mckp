# Certifying robust pricing MCKP — v3 reproducibility code

This reviewer-facing repository contains the solver, exact certificate checker,
tests, experiment drivers, and released computational evidence for the candidate
v3 revision of the working paper currently available as
[arXiv:2603.18653](https://arxiv.org/abs/2603.18653),
*Exact Endpoint Protection and Verifiable Optimality Bounds for Gamma-Robust
Finite-Menu Pricing*. The live arXiv record retains the v2 title until the
manuscript update is posted.
The paper-aligned code and evidence tag is
[`v3.1.0`](https://github.com/eric939/certifying-robust-pricing-mckp/tree/v3.1.0).

The manuscript, submission packages, publication notes, and historical paper
sources are intentionally not part of this software repository.

## What is included

- `src/robust_mckp/`: the finite-menu robust-pricing/MCKP model, full
  binary64-distinct threshold construction, hull relaxation and rounding,
  exact branch-and-bound, parametric sweep, direct robust-feasibility check,
  and exact full-family primal-dual certificate with optional LP-to-rounding
  provenance.
- `tests/`: correctness and adversarial regression tests.
- `experiments_nested/` and `experiments_case_retail/`: reproducible instance
  generators and experiment utilities.
- `scripts/`: experiment, mathematical-audit, benchmark, and summarization
  drivers.
- `results/release/2026-08-28-paper-a-v3/`: the canonical v3 evidence release:
  135 exact certificate intervals for 45 controlled instances, three exact
  pilot certificates, derived pilot records, protocols, provenance, source
  snapshot, exact binary64 input snapshots, and nested SHA-256 manifests.
- `results/release/2026-08-29-paper-a-v3-independent-rq/`: the frozen
  separated-reference/cost extension: mixed-sign favorable and adversarial
  instances, schema-v2 exact rounding witnesses, an all-integer-Gamma
  certificate frontier, scaling diagnostics, and split-calibration controls.
- `results/release/2026-08-09-paper-a-final-r2/`: the frozen baseline release
  from which the v3 controlled instances were reconstructed.

## Install and verify

```bash
uv sync --extra experiments --extra validation --extra dev
make verify PYTHON=.venv/bin/python
```

This runs the complete test suite, checks all dated releases byte-for-byte,
reconstructs the frozen instances, independently verifies the exact certificates,
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

## Optional LP-to-rounding provenance

`build_global_certificate(..., include_rounding_provenance=True)` emits a
schema-version-2 certificate. In addition to the complete family of sharp
fixed-threshold dual witnesses, it records at a threshold attaining the global
LP upper endpoint:

- one exact basic LP position per group;
- the sole fractional group, adjacent upper-hull endpoints, and rational
  mixing weight (or null metadata for an integral optimum);
- the literal lower-endpoint rounded selection; and
- exact primal cost/value, rounded value, loss, and adjacent value jump.

`verify_global_certificate` reconstructs every transformed coefficient and
upper hull from the original instance. It checks LP feasibility and basic
structure, primal-dual equality, attainment of the global LP maximum, direct
sorted-Gamma feasibility of the rounded selection, the adjacent-jump identity,
and that the certificate's `H` dominates the rounded value. Version-1
certificates remain accepted unchanged; the optional field is never inferred
for a legacy certificate.
