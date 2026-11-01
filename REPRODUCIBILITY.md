# Reproducibility

## Environments

- Library and tests: `uv sync --extra dev --extra milp --extra scip` (the
  checked-in `uv.lock` freezes the dependency graph).
- The revised paper's campaigns ran on Python 3.12.12 (macOS arm64) with the
  package versions in `studies/paper-a-revision/requirements-study.txt`.
  `make study-env` creates `.venv-study` with exactly these. The tables are
  reproduced byte for byte with other versions too; the figure PDF only with
  the recorded matplotlib.

## Revised paper (v4): verify the released evidence

```bash
make study-env
make verify-evidence PYTHON=.venv-study/bin/python       # a few minutes
make verify-evidence-full PYTHON=.venv-study/bin/python  # + portfolios and re-runs
```

`studies/paper-a-revision/verify_evidence.py` checks, for both protocols:

1. every file in `MANIFEST.sha256`, except those the public tree omits because
   they are regenerated: the protocol-v1 base portfolios (`instances/`,
   139 MB) and the generated LaTeX tables and figure (`reporting/*.tex`,
   `reporting/*.pdf`, kept out of the tree with the other manuscript files);
2. the frozen copies of protocol and code against `FROZEN_SOURCES.sha256`;
3. that `planned_tasks.json` equals the task list of `study/protocol.py` and
   `study/protocol_v2.py`, with exactly one raw record per planned run;
4. that every derived file (per-run CSVs, summaries, all LaTeX tables and the
   figure) rebuilt from the raw records alone equals the manifest byte for byte;
5. (`--instances`) that all 66 + 24 base portfolios regenerated from their
   seeds equal their recorded hashes;
6. (`--rerun K`) that K sampled runs on small instances, repeated with the
   library, reproduce status, objective, bound and original menu IDs exactly
   (MILP solvers: objective within the 1e-6 tolerance). Runs that ended at a
   time limit are not sampled.

## Revised paper (v4): re-run a campaign

Use new, empty evidence directories; never write into a dated release.

```bash
cd studies/paper-a-revision && source env.sh
PY=../../.venv-study/bin/python
# protocol v1: freeze, base portfolios, validation panel, 1,764 runs
$PY -m study.run_campaign --evidence /path/to/new-v1 --stage freeze
$PY -m study.run_campaign --evidence /path/to/new-v1 --stage instances
$PY -m study.run_campaign --evidence /path/to/new-v1 --stage validate
$PY -m study.run_campaign --evidence /path/to/new-v1 --stage run --workers 3
# protocol v2 reads the v1 portfolios from PAPER_A_V1_EVIDENCE
export PAPER_A_V1_EVIDENCE=/path/to/new-v1
$PY -m study.run_campaign_v2 --evidence /path/to/new-v2 --stage freeze
$PY -m study.run_campaign_v2 --evidence /path/to/new-v2 --stage instances
$PY -m study.run_campaign_v2 --evidence /path/to/new-v2 --stage robustness
$PY -m study.run_campaign_v2 --evidence /path/to/new-v2 --stage run --workers 3
# tables, macros, supplement and figure
$PY -m study.report /path/to/new-v1 /path/to/generated /path/to/figures
$PY -m study.report_v2 /path/to/new-v2 /path/to/generated /path/to/figures
```

Protocol v1 takes about a day on three workers (Panel L has a 600 s limit);
protocol v2 a few hours. To re-run against the released portfolios instead of
regenerated ones, first store them with
`python verify_evidence.py --instances --write-instances`. Objectives, bounds
and decisions of runs that finish within their limit are reproducible;
running times are not.

`make reproduce-smoke` runs the validation panel (180 small instances, every
method against brute force and the exact certificate). The released
`validation.json` records 179 of 180 checks passing; the one failure is the
v3 reference-solver defect fixed in v4 (`VALIDATION_NOTE.md`), so a v4 run
passes all 180.

## v3 releases

The v3 evidence was produced by tag `v3.1.1`; its verifiers pin the hashes of
the v3 sources and therefore run against that tag.

### Verify the released evidence

```bash
make verify-archive PYTHON=.venv/bin/python   # from a v4 checkout
make verify PYTHON=.venv/bin/python           # inside a v3.1.1 checkout
```

The verification target checks:

1. the complete test suite;
2. the frozen baseline release and its scientific gates;
3. both nested v3 manifests and every released byte;
4. all 135 controlled full-family certificates against frozen exact binary64
   instance snapshots, without platform-dependent generator reconstruction;
5. all three pilot certificates after reconstruction from the public derived
   model record; and
6. the separated-reference/cost manifest, schema-v2 exact certificates and
   rounding witnesses, calibration implication audit, and integer-Gamma
   frontier; and
7. the absence of tracked manuscript, build, and restricted-data artifacts.

It does not require manuscript source, a TeX system, or the Dominick's raw
files.

### Fresh bounded reproduction

```bash
make reproduce-smoke PYTHON=.venv/bin/python
```

This writes only to `results/local/`: it runs a reduced mathematical audit and
then regenerates a reduced solver/certificate/scalability release from that
audit. The canonical released numbers remain immutable.

### Full evidence regeneration

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

### Rebuild the separated-reference/cost campaign

The campaign uses fixed seeds and refuses to overwrite its destination. Run it
into a new local directory, then verify every released hash and exact
certificate from the stored binary64 inputs:

```bash
.venv/bin/python scripts/run_v3_independent_rq_campaign.py \
  --output-dir results/local/v3-independent-rq
.venv/bin/python scripts/verify_v3_independent_rq_release.py \
  --release-dir results/local/v3-independent-rq
```

Its favorable/adversarial study and integer-Gamma frontier use schema-v2
rational certificates. The larger scaling table is a floating diagnostic. The
calibration study separates tuning, calibration, and one-step evaluation;
exchangeable correlated and heavy-tail cells are positive controls, while
AR(1), change-point, and menu-dependent cells are assumption stress tests.

### Rebuild the v3 controlled certificate study

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

### Rebuild the exploratory Dominick's pilot

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
