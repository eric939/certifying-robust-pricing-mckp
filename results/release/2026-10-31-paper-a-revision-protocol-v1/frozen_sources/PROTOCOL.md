# Paper A TOP revision — computational study protocol (v1)

Frozen 31 October 2026, before any study instance was generated. The design is
the pure function `study/protocol.py::tasks()`; the frozen source files and
their SHA-256 hashes are recorded in `FROZEN_SOURCES.sha256` in the evidence
directory. Source files must not change after freezing; a change requires a new
protocol version and a new evidence directory.

Purpose: answer Reviewer 1 item 5 and Reviewer 3 items 3–5 of TOPR-D-26-00131
(method and component comparison, sensitivity, scalability, complete tables with
dispersion and failure counts, hull fraction by menu family, and the menu statistic
`Δ^v_max / V̲` for every instance).

## Model and input path

* Relative-residual model of §3.3: widths `δ⁻ = ϱĝ`, `δ⁺ = ϱĝ/(1−ϱ)`; integer Γ.
* Separate unit costs `q_i ≠ a_i`; `a_i` anchors the admissible band only.
* Coefficients come from the corrected exact primitive-input path of the
  repository (local commit `75d4704`, `ExactPrimitivePricingData`): exact
  rational admissibility `|x−a| ≤ σa` on the binary64 inputs, original menu IDs
  retained, exact rational `v, s, t`. The historical generator and the
  EPS-filtered `from_pricing_data` path are **not** used.
* Solvers use the correctly rounded binary64 values of the exact coefficients.
  Every returned decision is mapped to original option IDs and re-checked in
  exact rational arithmetic (admissibility of the original ID, robust residual
  `Res_Γ ≥ 0`, revenue). A decision failing this check is a failure.

## Generator (study salt 20261031; pilot salt 20261030 is excluded)

Per product: `a ~ U[10,60]`, `q = a·U[0.55,0.85]`, `ω ~ U[0.5,2]`, elasticity
`ε ~ U[0.6,2.5]`, scale `D ~ U[50,500]`, band `σ = 0.2`, demand
`ĝ(x) = D (x/a)^(−ε) · mult`. Families:

* **regular**: m equally spaced prices across the band, generated boundary
  prices moved inward by `inward_generated_price`; `mult = 1`.
* **charm** (irregular price endings): m distinct prices drawn from the
  ladder `{k + e : e ∈ {.09,.19,…,.99}}` inside the band (exact admissibility);
  left-digit demand bonus `mult = 1.06` for ending .99, `1.03` for .49.
* **noisy**: regular grid, `mult = exp(N(0, 0.08²))` per option (estimated,
  possibly non-monotone demand).

Margin target `κ = κ_lo + 0.5(κ_hi − κ_lo)`, where `κ_lo` is the nominal ratio
of the revenue-maximizing plan and `κ_hi` the largest nominally feasible target
(bisection). Seeds: `SeedSequence([20261031, panel_code, family_code, n, m, rep])`.

## Panels

| Panel | Families | (n, m) | Portfolios | Γ | ϱ | Limit |
|---|---|---|---|---|---|---|
| V (validation) | 3 | (6,4), (8,4), (10,3) | 2 | {0, ⌊√n⌋, ⌊0.1n⌋, ⌊0.5n⌋, n} | {0.1, 0.3} | 60 s |
| S (sensitivity, comparison) | 3 | (200,16) | 4 | same five | {0.05, 0.10, 0.20, 0.30} | 60 s |
| L (scale) | regular, charm | n ∈ {1000,2000,5000} × m ∈ {8,16,32} | 3 | ⌊√n⌋ | 0.10 | 600 s |

Panel S: 240 runs × 6 methods = 1,440 tasks. Panel L: 54 runs × 6 methods =
324 tasks. Panel V: 180 runs, every method plus brute force, the repository's
reference `solve()` and its exact rational global certificate.

## Methods (all single-threaded)

* **HR** — Algorithm 1: complete threshold family `B = {0} ∪ {t_ij}` (exact
  binary64 deduplication), canonical hulls rebuilt at every threshold,
  hull-greedy LP by weighted selection, round-down and round-up+repair, then
  completion; direct robust check of each improving candidate. Records `H`
  (repair+completion) and `H_rd` (round-down only) in the same run.
* **HR-sweep** — as HR, but a group's hull is reused when Proposition 23 shows
  its transformed point set is unchanged (enumeration ablation).
* **HR-dom** — HR-sweep after threshold-uniform dominance (Proposition 19)
  (preprocessing ablation).
* **HR+BB** — HR-sweep, then the exact layer: feasible thresholds in
  decreasing LP order; a threshold is pruned when its LP value is at most the
  incumbent plus the tolerance; otherwise depth-first fixed-θ branch-and-bound
  with integer-safe dominance (Definition 17), residual hull-greedy LP bounds and
  branching on the fractional group.
* **DNC** — threshold-interval divide and conquer in the style of Hansknecht
  et al. (2018) and Büsing, Gersing and Koster (2023): for `θ ∈ [b_l, b_r]`,
  `Σ s^{b_r} ≥ Γ b_l` is a valid relaxation whose hull-greedy LP bounds the
  interval; best-bound interval processing, midpoint splits with rounding
  incumbents, single thresholds solved by the same fixed-θ B&B. Not warm-started
  by HullRound. The authors' code (general binary programs via a commercial MIP
  solver) is not used; see the manuscript for the reason.
* **MILP** — compact Bertsimas–Sim counterpart
  `max Σ v z : Σ s z − Γθ − Σ π ≥ 0, π_i + θ ≥ Σ_j t_ij z_ij, Σ_j z_ij = 1`,
  HiGHS 1.15.1 via highspy, `threads=1`, `mip_rel_gap=1e-6`, `mip_abs_gap=0`,
  `random_seed=0`. SCIP and Gurobi are not installed/licensed and are not used.

Optimality tolerance for every exact method: relative `10⁻⁶`.

## Outcomes and statuses

Every planned task is retained: `CERTIFIED` (HR family), `OPTIMAL`,
`TIME_LIMIT` (with incumbent and finite bound), `TIME_LIMIT_NO_INCUMBENT`,
`INFEASIBLE`, `ERROR`, `KILLED` (hard kill at limit + 300 s), `MISSING`.
An incumbent failing the exact rational check counts as a failure.
Time-limited HullRound reports Theorem 21 records (evaluated LP values and the
trivial bound `Σ_i max_j v_ij`).

Reported per configuration: run count; termination counts; own gap
`(U−L)/U`, deficit to the best exactly verified revenue of the run, and time
(median, IQR, maximum); hull fraction by family; `Δ^v_max / V̲` per instance
(CSV) with the Corollary 15(a) check `(U−H)/U ≤ min{1, Δ^v_max/V̲}`.

## Execution

`guard.py run --expected-gb 1 -- <env python> -m study.run_campaign ...`, three
concurrent task subprocesses, small configurations first. The machine is
shared (Apple M4, 10 cores, 16 GB); load averages are recorded per task.
Wall-clock and CPU time are both recorded; times are descriptive.

## Pilot (excluded from the evidence)

Pilot salt 20261030 was used only to check implementation correctness against
brute force and to choose time limits. At n=5000, m=32 (160,001 thresholds)
HR-sweep took 369 s, HR 498 s, DNC 6 s and the MILP 409 s on one pilot
instance (charm family); the 600 s limit for Panel L was set from these
observations. Pilot Panel-S configurations finished in under 2 s per method,
and 72 + 180 tiny pilot instances agreed with brute force. After the pilot, an
untimed warm-up call on a 4-product instance was added to every task so that
JIT-cache loading is not timed.
