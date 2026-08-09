# Paper A claim/evidence matrix

Status values: `pending`, `verified`, `revised`, `removed`, `blocked`.

| ID | Claim | Kind | Required proof/evidence | Status |
|---|---|---|---|---|
| A-M1 | Admissible finite-menu ratio-margin pricing is exactly an MCKP after the baseline--slack transformation. | theorem | Bidirectional algebra; input assumptions; brute-force model parity. | verified |
| A-M2 | Integer-budget demand uncertainty gives the sorted-Gamma protection term. | theorem | Robust-counterpart proof; Gamma edge cases; brute-force uncertainty oracle. | verified |
| A-M3 | The protection term has the scalar threshold dual representation. | theorem | LP dual derivation and strong-duality assumptions. | verified |
| A-M4 | A robust-feasible optimum is represented by zero or an original admissible deviation. | theorem | Complete piecewise-linear breakpoint proof; Gamma=0/n and ties. | verified |
| A-M5 | A fixed-threshold problem is exactly an MCKP with nonnegative transformed costs. | theorem | Algebra; infeasible-capacity and baseline edge cases. | verified |
| A-M6 | Upper-hull segment greedy solves the fixed-threshold LP relaxation. | classical theorem | Self-contained proof plus primary MCKP sources; LP oracle parity. | verified |
| A-M7 | Round-down is feasible and loses no more than one adjacent hull-value jump. | theorem | Explicit rounding proof; randomized/adversarial property tests. | verified |
| A-M8 | The fixed-threshold LP--IP gap is bounded by the maximum item hull jump. | theorem | Derive from a feasible rounded solution; exhaustive small instances. | verified |
| A-M9 | The relative gap is O(1/n) under uniform jump and linear-growth assumptions. | conditional corollary | Explicit sequence-of-instances assumptions; no unconditional wording. | revised |
| A-M10 | Fixed-threshold branch-and-bound is exact under complete branching and valid LP pruning. | theorem | Search-partition proof; brute-force parity including below-hull optimum. | verified |
| A-M11 | Complete threshold search is globally exact; limited runs have a valid gap only with a record for every threshold. | theorem | Max-over-family proof; tests for completed and missing-record states. | verified |
| A-M12 | Guarded sweep has the same validity as independent reconstruction. | proposition | Exact-arithmetic invariant; binary64 parity/regression tests. | verified |
| A-I1 | Implementation preserves every binary64-distinct original deviation plus zero. | implementation | Unit and unique-feasible close-threshold regression tests. | verified |
| A-I2 | Every returned robust incumbent is checked by direct sorted-Gamma evaluation. | implementation | Source trace plus tests with rejected near-infeasible candidates. | verified |
| A-I3 | Hulls are LP-bound objects and exact search retains non-dominated below-hull options. | implementation | Source trace plus targeted integer-optimum test. | verified |
| A-E1 | HullRound certificate holds throughout the registered experiment grid. | observation | Canonical raw rows and zero certificate failures. | verified |
| A-E2 | Sweep and enumeration agree on registered validation rows. | observation | Canonical parity fields and zero unexplained mismatches. | verified |
| A-E3 | Exact mode agrees with brute force/open solvers on matched completed rows. | observation | Canonical matched results and availability disclosure. | verified |
| A-E4 | Sweep reduces fixed-threshold construction work on tested families. | observation | Prespecified construction metrics; no solver-dominance wording. | removed |
| A-E5 | Controlled pricing instances display a revenue--protection tradeoff. | observation | Dated semi-synthetic protocol/raw results; no causal or field claim. | removed |
| A-L1 | MCKP LP geometry and exact methods are classical. | literature | Primary original papers/publisher metadata with locators. | verified |
| A-L2 | Bertsimas--Sim budgeted uncertainty supports the stated protection model. | literature | Original paper/publisher metadata and exact formulation locator. | verified |
| A-L3 | Paper A's combined pricing/full-breakpoint/certification contribution is distinct from closest work. | qualified novelty | Reproducible search log and primary-source comparison table; qualified language. | verified |
| A-B1 | Paper B begins from A's threshold family but contributes simultaneous family bounds absent from A. | scope/lineage | Side-by-side theorem and algorithm map; final cross-citation. | verified |
