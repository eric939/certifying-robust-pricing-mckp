# Reproducibility

The only evidence supporting the final Paper A manuscript is
`results/release/2026-08-09-paper-a-final-r2/`. Earlier local outputs and legacy
experiment drivers are not part of the paper's evidence.

## Environment

```bash
uv sync --extra experiments --extra validation --extra dev
```

The lockfile `uv.lock` fixes the Python dependency resolution. The release
protocol records the operating system, Python, NumPy, SciPy, git commit, seeds,
and numerical policies.

## Verify the frozen release

```bash
make verify PYTHON=.venv/bin/python
```

This command checks:

1. all unit and regression tests;
2. the release SHA-256 manifest;
3. the mathematical-oracle audit and every experiment gate;
4. equality of manuscript macros/figure with the canonical release copies;
5. public and anonymous LaTeX builds;
6. missing references, citations, files, and overfull layout boxes; and
7. absence of author, institution, repository, and arXiv identity in the
   anonymous PDF.

## Regenerate in a new dated directory

Never overwrite the frozen directory. A protocol or algorithm change requires
a new date/name and a new manuscript revision.

From a clean source commit:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mathematical_audit.py \
  --cases 500 --lp-cases 50 --sweep-cases 30 \
  --output /tmp/paper-a-mathematical-audit.json

PYTHONPATH=src:. .venv/bin/python scripts/run_paper_a_release.py \
  --audit-json /tmp/paper-a-mathematical-audit.json \
  --output-dir results/release/NEW-DATED-DIRECTORY
```

Then copy only the generated macro and figure into `paper/current/generated/`
and `paper/current/figures/`, respectively, and run the full verification gate.
The canonical CSV/JSON files, not prose, are the source of every reported
number.

## Interpretation policy

- Theorems are exact-arithmetic statements.
- Exhaustive and independent-solver comparisons are implementation checks.
- The binary64 tolerance is recorded and is not interval arithmetic.
- Timing is observational and hardware dependent.
- Synthetic pricing instances are controlled structural tests, not consumer,
  causal, or commercial evidence.
- Every threshold set retains zero and every binary64-distinct original
  deviation; structural candidates are never tolerance-clustered.
