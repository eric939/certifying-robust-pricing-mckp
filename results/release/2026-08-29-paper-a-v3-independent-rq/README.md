# Paper-A v3 independent-reference/cost campaign

Campaign ID: `2026-08-29-paper-a-v3-independent-rq`.

This is a deterministic full evidence campaign frozen before manuscript use. It is not a preregistration.

## What this release tests

- Reference prices `r_i` and unit costs `q_i` are separate and nonidentical; `r_i` controls menu admissibility and `q_i` controls the ratio margin. This is parameter separation, not a claim of stochastic independence.
- Every exact-certificate instance contains both negative- and positive-surplus menu options.
- Favorable instances satisfy bounded-group intuition; dominant-group instances deliberately include one large group.
- Exact schema-v2 rational certificates verify the upper endpoint, incumbent, literal LP round-down witness, and `U-H <=` literal rounding loss. Larger scaling rows are numerical diagnostics, not exact certificates.
- Split calibration is tested under two exchangeable positive controls. AR(1), change-point, and menu-dependent cells are assumption stress tests and carry no coverage guarantee.

## Released artifacts

- `certificate-study.csv`: 12 exact completed intervals and HiGHS cross-checks.
- `certificate-study/instances/`: authoritative binary64 snapshots.
- `certificate-study/certificates/`: independently checkable exact certificates.
- `scaling-study.csv`: 16 larger numerical full-family sweeps.
- `gamma-frontier.csv`: 21 exact integer-budget frontier points.
- `calibration-study.csv`: 600 independent tune/calibrate/test repetitions.
- `summary.json`, `summary-macros.txt`, `figures/`, `protocol.json`, and `manifest.json`.

## Reproduce and verify

From a clean checkout with the locked environment installed:

```bash
.venv/bin/python scripts/run_v3_independent_rq_campaign.py --output-dir results/local/v3-independent-rq
.venv/bin/python scripts/verify_v3_independent_rq_release.py --release-dir results/local/v3-independent-rq
```

The generator refuses to overwrite an existing directory. Binary64 snapshots, rather than regenerated transcendental inputs, are authoritative for exact verification.

## Limits

This is a controlled/semi-synthetic study, not causal evidence of market response or field deployment. The larger scaling table is numerical. Monte Carlo rates have finite sampling uncertainty, shown with Wilson intervals in `summary.json`. Negative controls diagnose assumptions; they do not estimate real-world failure frequencies.
