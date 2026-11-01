# Re-check of the submitted Paper A evidence (31 Oct 2026)

Release: `paper-a/code/results/release/2026-08-09-paper-a-final-r2` (macros of
`main_submitted.tex` §7). Source: public checkout files at 3df022a (the release
commit b65e566 is not in the local object store; all 114 instances and HullRound
values reproduce bit-for-bit, so the generator and solver are unchanged). Run:
`python check_submitted.py SRC RELEASE_DIR` (read-only; public repo unchanged).

1. Admissibility. The historical generator filters on the shock |u| ≤ σ + EPS
   and then on float bounds (1±σ)a ± EPS. Over the 54 distinct (seed, n, m)
   prefixes, 28,356 options are kept; 0 of them fail the exact test
   |x − a| ≤ σa, and 0 rejected options pass it (5 shocks lie within 1e-6 of σ,
   none decisive). No submitted instance contained an inadmissible option.
2. Rounding defect of the reference heuristic (VALIDATION_NOTE.md). In 88 of
   114 rows some threshold had more than one group with 0 < λ < 1 (spurious
   λ ≈ 1e-16), but the returned H changed in only 5 rows when the fractional
   group is taken from the greedy fill: 2 exact-comparison rows (H lower by
   18.5 and 1.29, i.e. the defect happened to help) and 3 scalability rows (H
   higher by 0.21–0.55; objective values there are not reported). No
   certificate-study row changed. Effect on reported macros: HullRound median
   gap to the optimum 0.00787 % → 0.00851 % (max 0.108 % unchanged); all other
   macros (exact agreement, feasibility counts, loss ratio, LP difference,
   hull fraction, timings) do not depend on it. Not re-checked: the round-down
   loss L_rd per row (its ≤ bound assertion passed in the release).
