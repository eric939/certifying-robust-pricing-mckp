# robust_mckp — certified solvers for Γ-robust finite-menu pricing

`robust_mckp` chooses one price from a finite menu for each product so that
revenue is maximal while a margin requirement holds even when up to Γ products
take their worst-case deviation. It is the multiple-choice knapsack problem
(MCKP) with a Bertsimas–Sim budget, and the package solves it to optimality or
returns a decision with a **certified interval** `H ≤ OPT ≤ U`. Every returned
decision can be re-checked in exact rational arithmetic, and the global upper
bound can be emitted as an exact primal-dual certificate.

This repository is the code and evidence release of the paper *A Certifying
MCKP Framework for Γ-Robust Discrete Pricing* (Z. Y. E. Shao; revised version
under review). The arXiv record
[arXiv:2603.18653](https://arxiv.org/abs/2603.18653) holds an earlier version
under the title *Exact Endpoint Protection and Verifiable Optimality Bounds for
Gamma-Robust Finite-Menu Pricing*. Release
[`v4.0.0`](https://github.com/eric939/certifying-robust-pricing-mckp/tree/v4.0.0)
matches the revised paper; [`v3.1.1`](https://github.com/eric939/certifying-robust-pricing-mckp/tree/v3.1.1)
matches the submitted version.

## Install

```bash
pip install "certifying-robust-pricing-mckp @ git+https://github.com/eric939/certifying-robust-pricing-mckp@v4.0.0"
```

Python 3.10 or later. The core needs NumPy and Numba. Optional extras:
`[milp]` (HiGHS), `[scip]` (SCIP), `[study]` (both MILP solvers and
matplotlib, to reproduce the paper's study), `[dev]` (tests, build).
The import name is `robust_mckp`.

## Quick start

```python
import robust_mckp as rm

# Option j of group i: revenue v, nominal margin surplus s, endpoint loss t >= 0.
# A decision is feasible if  sum_i s_i - (sum of the Gamma largest t_i) >= 0.
inst = rm.instance_from_arrays(
    values=[[10, 12, 13], [8, 9, 11], [5, 7, 8]],
    margins=[[4, 2, 1], [3, 2, 0], [2, 1, -1]],
    losses=[[1, 2, 3], [1, 1, 2], [0.5, 1, 2]],
    gamma=1,
)

res = rm.solve(inst)                       # optimal decision (threshold DnC)
print(res.status, res.selection, res.revenue, res.exact.feasible)

res = rm.solve(inst, method="certified")   # HullRound's certified interval, fast
print(res.revenue, "<= OPT <=", res.upper_bound, "gap", res.gap)

rm.verify(inst, res.selection)             # exact rational re-check
cert = rm.certify(inst, res.selection)     # exact global certificate (Theorem 12)
assert rm.verify_global_certificate(inst, cert).valid
```

From pricing primitives (reference prices, unit costs, menus, demand
estimates and their deviations), build the instance with
`rm.from_pricing_data(...)`; menus are filtered to admissible prices and
`Result.original_ids` maps the decision back to the supplied menus. For
inputs that must be handled in exact rational arithmetic end to end, pass
`rm.ExactPrimitivePricingData(...)` directly to `solve`, `verify` and
`rm.build_primitive_certificate`. See [`examples/`](https://github.com/eric939/certifying-robust-pricing-mckp/tree/v4.0.0/examples).

### Methods

`rm.solve(instance, method=..., time_limit=None)`:

| `method` | Paper | Returns |
|---|---|---|
| `"exact"` (default) | threshold divide and conquer with interval records | optimal decision |
| `"certified"` | HullRound with interval records (Proposition 23) | decision with `H ≤ OPT ≤ U` |
| `"hullround"` | HullRound, Algorithm 1, guarded sweep (Proposition 24) | decision with `H ≤ OPT ≤ U` |
| `"hullround-full"` | HullRound, rebuilding every hull | decision with `H ≤ OPT ≤ U` |
| `"hullround-dominance"` | guarded sweep after universal dominance (Proposition 19) | decision with `H ≤ OPT ≤ U` |
| `"exact-layer"` | HullRound followed by the exact full-breakpoint layer | optimal decision |
| `"milp-highs"`, `"milp-scip"` | compact robust MILP | optimal decision |
| `"bruteforce"` | enumeration (tiny instances) | optimal decision |
| `"reference"` | the v3 pure-Python HullRound | feasible decision |

`solve` returns a `Result` with `status` (`optimal`, `certified`,
`time_limit`, `infeasible`), `selection` (one option index per group),
`original_ids`, `revenue`, `upper_bound`, `gap`, `elapsed`, `exact` (the
exact check) and `details` (thresholds, nodes, interval evaluations, ...). On a
time limit the best decision and a valid upper bound are returned. Wrap an
instance in `rm.FlatInstance` to reuse its arrays across calls.

**What the guarantees are.** The solvers compute in binary64. "Optimal" means
optimal up to a relative tolerance of 1e-6, and HullRound's `H` and `U` are
binary64 values. The exact check (`Result.exact`, on by default) recomputes the
robust residual and revenue of the returned decision in rational arithmetic,
and `rm.certify` builds an upper bound that holds exactly. A decision that
fails the exact check raises a `RuntimeWarning`; in the paper's study this
happened for 5 of the 390 SCIP decisions and for no other method.

The first call of a compiled method in a new process includes Numba's
compilation (a few seconds); the compiled code is cached on disk afterwards.

## Reproducing the paper

The revised paper's study ran two protocols, each fixed before its instances
were generated: 1,764 runs (protocol v1) and 1,260 runs (protocol v2). Both are
released under [`results/release/`](https://github.com/eric939/certifying-robust-pricing-mckp/tree/v4.0.0/results/release) with protocols, frozen
sources, environment, one raw record per run and SHA-256 manifests. The study
code is in [`studies/paper-a-revision/`](https://github.com/eric939/certifying-robust-pricing-mckp/tree/v4.0.0/studies/paper-a-revision).

```bash
git clone https://github.com/eric939/certifying-robust-pricing-mckp
cd certifying-robust-pricing-mckp
make study-env                      # the campaign's exact package versions
make verify-evidence PYTHON=.venv-study/bin/python
```

This checks every released byte and that the task lists equal the protocol
code, and it rebuilds every table, CSV, summary and the figure of the paper
from the raw records, byte for byte. `make verify-evidence-full` also
regenerates all base portfolios from their seeds and re-runs a sample of runs,
which must reproduce status, objective, bound and decision exactly. See
[REPRODUCIBILITY.md](https://github.com/eric939/certifying-robust-pricing-mckp/blob/v4.0.0/REPRODUCIBILITY.md) for complete re-runs of either campaign.

`make verify` runs the test suite, the evidence check and the check of the v3
releases.

## Repository layout

- `src/robust_mckp/`: the library. `api.py` (`solve`, `verify`, `certify`),
  `fast/` (compiled kernels and solver drivers of the paper's study), `flat.py`
  (flat arrays and the exact check), `global_certificate.py` and
  `primitive_certificate.py` (exact certificates), and the v3 pure-Python
  modules (`solver`, `exact_bnb`, `parametric_sweep`, ...).
- `studies/paper-a-revision/`: protocols, campaign drivers, report generators
  and the evidence verifier of the revised paper.
- `results/release/`: dated evidence releases (see its README).
- `tests/`: tests of the library and of the v3 releases.
- `scripts/`, `experiments_nested/`, `experiments_case_retail/`: drivers of
  the v3 experiments.
- `examples/`: short scripts using the library.

## Earlier releases

The v3 evidence (`results/release/2026-08-*`) was produced by `v3.1.1`. Its
verifiers pin the hashes of the v3 sources, so they run against that tag:
`make verify-archive` checks out `v3.1.1` in a temporary git worktree and runs
all three. The v3 drivers in `scripts/` still run on v4, where the v3 solver is
`solve_reference`.

Changes from v3 are listed in [CHANGELOG.md](https://github.com/eric939/certifying-robust-pricing-mckp/blob/v4.0.0/CHANGELOG.md). One concerns the
v3 reference solver: its rounding step could round a group with a spurious
fractional weight of order 1e-16 instead of the true fractional group. This is
fixed in v4. The decision was always feasible and checked; the effect on the
submitted results is reported in the revised paper (§7.2).

## Exact certificates

`build_global_certificate(instance, selection, include_rounding_provenance=True)`
(or `rm.certify`) emits a schema-version-2 certificate. In addition to the
complete family of sharp fixed-threshold dual witnesses, it records at a
threshold attaining the global LP upper endpoint:

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

## Citation and license

Please cite the paper and the software; see [CITATION.cff](https://github.com/eric939/certifying-robust-pricing-mckp/blob/v4.0.0/CITATION.cff).
MIT licensed. No manuscript source, submission package or restricted raw
dataset is part of this repository; the Dominick's pilot of v3 publishes only
derived records and source-file hashes.
