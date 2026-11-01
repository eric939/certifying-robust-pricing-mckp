# Protocol v2 — TOP revision computational study (Paper A)

Written and hashed on 31 October 2026 before any v2 study instance or v2 run.
Protocol v1 (`PROTOCOL.md`, evidence `2026-10-31-protocol-v1`) is unchanged and
its evidence directory is only read. v2 evidence:
`/Volumes/LaCie/Archives/Codex/Projects/robust-mckp/Artifacts/paper-a-top-revision-study/2026-10-31-protocol-v2`.
The design is the pure function `study/protocol_v2.py:tasks()` (1,260 tasks).

## Questions

1. Does HullRound with interval records (method PIPE) return the same certified
   interval [H, U] as complete-family HullRound, and at what cost, on Panels S,
   L and H?
2. How does a second MILP solver (SCIP, via PySCIPOpt) compare with HiGHS on
   the compact formulation (Panels S, L, H)?
3. On instances built to stress the certificate (Panel H), how large are the
   certified gaps and how long do the exact methods take?
4. Price of robustness on Panel S: Z_rob(Γ, ϱ)/Z_nom and out-of-sample margin
   violation rates of robust and nominal decisions.

## Methods (single-threaded; relative optimality tolerance 1e-6 as in v1)

* HR, HR-sweep, HR+BB, DNC, MILP (HiGHS 1.15.1): as in v1 (`study/methods.py`, unchanged).
* PIPE (`study/methods_v2.py:run_pipeline`): intervals [b_l, b_r] of the
  threshold family processed best record first and split at the median, as in
  DNC, using the interval record of Proposition `prop:interval-record`. The
  split threshold of each processed interval and every single threshold are
  HullRound evaluations (hull LP, round-down, repair, completion). An interval
  whose record is ≤ H is discarded; a single threshold is not searched further.
  Output: certified [H, U], U = max(H, single-threshold LP values, discarded
  and open interval records). No branch-and-bound.
* MILP-SCIP (`run_milp_scip`): the v1 compact formulation, SCIP 10.0 via
  PySCIPOpt 6.3.0, limits/gap 1e-6, absgap 0, one thread, seed shift 0.
* DNC status relabelling (bug in v1 code, no v1 effect because no v1 run was
  infeasible): a DNC result labelled TIME_LIMIT_NO_INCUMBENT whose elapsed time
  is within the limit is an infeasible root and is recorded as INFEASIBLE with
  `status_relabelled = true`.

## Panels

* S and L: the 66 v1 base portfolios, read from the v1 evidence and
  hash-checked against its `instances_index.json`; the v1 (Γ, ϱ) grid; new
  methods PIPE and MILP-SCIP only; limits 60 s (S) and 600 s (L).
* H (`study/panel_h.py`, salt 20261101): n ∈ {20, 100, 500}, m ∈ {3, 4}, four
  portfolios each (24 bases); Γ ∈ {⌊√n⌋, ⌊0.5n⌋}; ϱ ∈ {0.2, 0.3}; seven methods;
  limit 300 s; 672 tasks. Generator: a ~ U[10,60], q = a·U[.55,.85],
  ω ~ U[.5,2], ε ~ U[1.5,3.5], demand scales 10^U[1,4] except product 0 whose
  scale is 10 × the largest other scale and whose menu has 3 prices; band
  σ = 0.4; equally spaced menus; margin target at τ = 0.85 between the
  revenue-maximizing and the largest nominally feasible ratio (v1: 0.5).
  Pilots (salt 20261030, not evidence) showed that τ ≥ 0.95 makes most
  instances infeasible and that feasible instances had certified gaps of
  0.01–0.8 % with exact methods finishing in < 1 s; the design is reported
  whatever it yields.

## Robustness evaluation (`study/robustness.py`)

For all 240 Panel S configurations: decisions are the v1 DNC decisions
(original menu IDs); Z_nom is the v1 DNC revenue at Γ = 0 of the same base.
10,000 draws per (base, ϱ), h_i i.i.d. U[−ϱ, ϱ], seeds
SeedSequence([20261101, base index, radius index]); realized demand by the
inverse residual map; a violation is a negative realized margin. Robust and
nominal decisions use the same draws.

## Execution and retention

Each task runs in its own subprocess with a kill at limit + 300 s; every
planned task gets a record (KILLED/ERROR included); records are never
overwritten. Three workers on the shared Apple M4; heavy stages run through the
storage-guard runner. Every returned decision is re-checked in exact rational
arithmetic (v1 `Encoded.exact_check`).

## Planned reporting

* Panel H: per method, termination counts, gap (median/IQR/max), deficit to
  best verified revenue, time (median/max); absolute Z and U.
* PIPE vs HR on S, L, H: counts of identical H and identical U; time ratio
  PIPE/HR (median, max); thresholds evaluated by PIPE / |B|.
* SCIP vs HiGHS on S, L, H: termination counts, time (median/max), agreement
  of objective values within tolerance.
* Robustness: per (Γ, ϱ), medians over 12 bases of Z_rob/Z_nom and of the
  violation rates of robust and nominal decisions.
