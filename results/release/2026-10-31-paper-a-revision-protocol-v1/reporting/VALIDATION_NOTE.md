# Validation panel V — the one non-passing check (recorded after the run)

Panel V (180 runs, study salt) passed 179/180 combined checks. The failing row:
`regular`, n=8, m=4, rep=1, Γ=⌊√n⌋=2, ϱ=0.1. Every exact method (exact
layer, DnC, HiGHS MILP) equals brute force (123003.028…); the study HullRound
returns H = 123003.028…, U equals the exact rational global certificate's U to
1e-16 relative, and the certificate of the returned decision verifies. Only the
check "study H equals the repository reference `robust_mckp.solve` H" fails:
the reference returns 122894.322….

Cause (diagnosed after the run, no study code changed): the reference
`greedy_lp` recovers each group's LP position from accumulated floating costs
by `searchsorted`. At θ ≈ 374.43 group 3 receives λ = 1.2e-16 instead of 0, and
`rounding._find_fractional_item` (first group with 0 < λ < 1) then rounds group
3 instead of the true fractional group 7 (λ = 0.422). The reference decision is
still robust feasible (it is checked), but its rounding step is not the one in
the manuscript at that threshold. The study implementation takes integer
positions directly from the greedy segment selection, so this cannot occur.

Consequence: none for the study's validity. It is a robustness defect in the
public reference implementation (v3.1.1 / local 75d4704), worth fixing there by
recording the fractional group during the greedy fill (as `greedy_lp` already
does in `fractional_item`) instead of re-deriving it from λ values.
