# Reproducibility

## Environment

The checked-in `uv.lock` freezes the dependency graph used by the release.

```bash
uv sync --extra experiments --extra validation --extra dev
```

## Verify the released evidence

```bash
make verify PYTHON=.venv/bin/python
```

The verifier checks the SHA-256 manifest, the independent mathematical audit,
all exact-solver comparisons, every robust-feasibility certificate, and every
one-item rounding bound. It does not require manuscript source or a TeX system.

## Fresh bounded reproduction

```bash
make reproduce-smoke PYTHON=.venv/bin/python
```

This writes only to `results/local/`: it runs a reduced mathematical audit and
then regenerates a reduced solver/certificate/scalability release from that
audit. The canonical released numbers remain immutable.

## Full evidence regeneration

Use a new, empty output directory; never overwrite a dated release.

```bash
PYTHON=.venv/bin/python
$PYTHON scripts/run_mathematical_audit.py \
  --output results/local/mathematical_audit.json
$PYTHON scripts/run_paper_a_release.py \
  --output-dir results/local/paper-a-release \
  --audit-json results/local/mathematical_audit.json
```

The audit validates the global robust optimum against independent oracles,
fixed-threshold LP bounds, the full binary64-distinct threshold family,
parametric-sweep reconstruction, and HullRound's robust certificate. Timing
observations are evidence, never proofs.
