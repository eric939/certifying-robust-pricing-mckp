# Paper A closest-work comparison

This table records what was identified in the stated formulation, abstract, or
cited sections of each primary source. “Not identified” is deliberately
narrower than “absent from the entire literature.” Exact publisher URLs,
locators, and bibliographic metadata are in `EVIDENCE_LEDGER_A.csv`.

| Source | Contribution identified in the reviewed source | Relationship to Paper A's claimed chain |
|---|---|---|
| Bertsimas and Sim (2004), REF-001 | General budgeted uncertainty and tractable robust counterparts, including discrete optimization | Supplies the classical uncertainty foundation and scalar auxiliary-variable logic; a finite-menu ratio-margin pricing-to-MCKP certificate chain is not identified. |
| Buesing, Gersing, and Koster (2023), REF-002 | Finite original-deviation family and branch-and-bound for general robust binary optimization with budget uncertainty | Directly prevents claiming deviation enumeration or family branching as new. Pricing reduction, MCKP hull rounding, and the pricing-form direct certificate are not the source's stated contribution. |
| Dyer (1984), Pisinger (1995), and Kellerer et al. (2004), REF-003/012/013 | Classical MCKP LP structure, algorithms, reduction, and exact methods | Supplies the hull/LP foundation. These sources do not study Paper A's demand-uncertain ratio-margin pricing model or its full robust-family accounting. |
| Nauss (1978), Ibaraki et al. (1978), and Sinha and Zoltners (1979), REF-010/011/004 | Early MCKP formulations and branch-and-bound/relaxation algorithms | Establishes that MCKP modeling and exact search are classical; no historical “first” claim is made by Paper A. |
| Monaci, Pferschy, and Serafini (2013), REF-014 | Exact solution of the Gamma-robust 0–1 knapsack problem | Closest robust-knapsack algorithmic foundation, but the reviewed formulation is not the exactly-one finite-menu pricing model or Paper A's MCKP rounding certificate. |
| Caserta and Voss (2019), REF-005 | Robust multiple-choice multidimensional knapsack under ellipsoidal uncertainty, with conic/linear reformulations and a matheuristic | Closest robust multiple-choice knapsack model found. Its stated uncertainty model and computational contribution differ from Paper A's integer-budget, scalar-threshold, pricing-specific certificate chain. |
| Harsha, Subramanian, and Ettl (2019), REF-007 | Practical omnichannel pricing with discrete prices and operational constraints | Establishes applied relevance of finite price menus and business constraints; budgeted demand robustness and the MCKP certification chain are not identified in the reviewed sections. |
| Thiele (2009), REF-009 | Robust multiproduct pricing under range forecasts and resource-use uncertainty | Establishes prior robust pricing. The reviewed formulation is not identified as the same finite-menu ratio-margin MCKP decomposition. |
| Hamzeei, Lim, and Xu (2022), REF-006 | Robust multiproduct pricing under interval uncertainty in price sensitivity | Establishes another robust-pricing line with a different uncertain object and model; Paper A does not claim robust pricing itself as new. |
| Shao, Mai, and Cheng (2026), REF-008 | Constrained logit-based pricing with price/probability constraints and MILP approximation | Covers richer substitutional demand and constrained pricing. Paper A instead studies separable own-price demand and a certifying Gamma-robust MCKP chain. |

## Feature-level conclusion

The comparison yields the following ownership boundary:

- Classical, not claimed alone: budgeted-robust scalar reformulation; original
  deviation candidates; MCKP modeling, hull geometry, LP solution, and exact
  search; robust pricing; robust knapsack.
- Paper A's scoped contribution: the end-to-end composition for one specific
  finite-menu ratio-margin pricing model, including its exact MCKP member
  transformation, pricing-form one-item certificate, complete original
  threshold accounting, exact-sign incumbent validation, and public
  failure-recording implementation.
- Outside Paper A: a simultaneous bound spanning many threshold intervals.
  That is the separate Paper B contribution.

Accordingly, Paper A is positioned as an applied algorithm-and-artifact paper
with a certifying composition, not as a foundational new theorem for general
robust optimization or MCKP.
