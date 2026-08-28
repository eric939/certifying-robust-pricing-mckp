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

The verification target checks:

1. all 122 tests;
2. the frozen baseline release and its scientific gates;
3. both nested v3 manifests and every released byte;
4. all 135 controlled full-family certificates against frozen exact binary64
   instance snapshots, without platform-dependent generator reconstruction;
5. all three pilot certificates after reconstruction from the public derived
   model record; and
6. the absence of tracked manuscript, build, and restricted-data artifacts.

It does not require manuscript source, a TeX system, or the Dominick's raw
files.

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

## Rebuild the v3 controlled certificate study

The v3 study reads the immutable baseline comparison table and always writes
to a new local directory:

```bash
.venv/bin/python scripts/run_v3_global_certificate_study.py \
  --release-dir results/release/2026-08-09-paper-a-final-r2 \
  --output-dir results/local/v3-global-certificate-study

.venv/bin/python scripts/freeze_v3_controlled_instances.py \
  --study-csv results/local/v3-global-certificate-study/global_certificate_study.csv \
  --output-dir results/local/v3-global-certificate-study/instances
```

Each of the 45 reconstructed instances has one shared sharp full-family LP
upper endpoint and three feasible lower-endpoint tracks. The resulting 135 JSON
files are checked in exact rational arithmetic.

The release additionally stores every controlled input coefficient as a Python
binary64 hexadecimal string. These snapshots are authoritative for portable
certificate verification because transcendental generator calculations can
differ by a final bit across numerical platforms.

## Rebuild the exploratory Dominick's pilot

Obtain `wcer.zip` and `upccer.csv` directly from the Kilts Center under its
terms, then run:

```bash
.venv/bin/python scripts/run_v3_dominicks_pilot.py \
  --movement-zip data/raw/dominicks/wcer.zip \
  --upc-csv data/raw/dominicks/upccer.csv \
  --output results/local/dominicks_cereal_pilot.json
```

Raw provider files and extracted transaction rows are ignored and must not be
committed. The UPC file is downloaded and hashed for provenance only; the
pilot computation does not read it. The released pilot status is
`failed_availability_and_unmodeled_dependence_gate`; that negative result is a
required reproducibility check, not a passing deployment claim.
