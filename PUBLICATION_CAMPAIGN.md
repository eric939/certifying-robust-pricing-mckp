# Paper A definitive-release campaign

## Objective

Produce the final, citable version of arXiv:2603.18653 and the exact public
software/evidence snapshot supporting it.  This release is a scientific freeze:
after the arXiv update, repository release, and archive tag agree on one commit,
Paper A is not extended with Paper B results.  Any later correction must be
identified explicitly as an erratum rather than silently changing the release.

## Scientific identity

Paper A is the pricing-and-decomposition paper.  Its primary object is a
finite-menu portfolio-pricing model with a ratio margin requirement under an
integer Bertsimas--Sim budget.  Its contribution chain is:

1. an exact reduction of the admissible pricing problem to MCKP;
2. an exact decomposition of the coupled robust constraint over zero and every
   original admissible option deviation;
3. fixed-threshold upper-hull LP bounds and robust-feasible one-item rounding
   with an additive certificate;
4. exact full-breakpoint branch-and-bound with a valid global anytime gap; and
5. a guarded parametric sweep that changes construction work without changing
   the threshold family or the certificate semantics.

Paper B starts from Paper A's full threshold family.  It asks how to bound that
family simultaneously, and owns the group-envelope interval bound, minimax
dominance, certified multiplier search, adaptive LP-family certification, and
their evidence.  Paper A must neither contain nor foreshadow those results as
its own.  Paper B must cite the frozen Paper A for the model and decomposition.

## Claim policy

Every material sentence is assigned one of four statuses:

- **Theorem:** proved under explicit assumptions and checked against exhaustive
  or adversarial finite instances where possible.
- **Implementation guarantee:** tied to a specific test and source locator;
  real-arithmetic and binary64 statements are distinguished.
- **Numerical observation:** regenerated from the dated canonical evidence
  release, with the instance family, seeds, environment, limits, and aggregation
  stated.  It is never described as proof.
- **Literature/context claim:** supported by a primary publisher, DOI registry,
  standards body, or original paper with an exact locator.  Negative novelty
  claims remain qualified.

Unsupported superlatives, general runtime superiority, causal pricing effects,
real-data validation, and solver-tolerance-as-proof language are prohibited.

## Mathematical gates

- The pricing-to-MCKP equivalence is checked in both directions, including
  denominator positivity, admissible-menu nonemptiness, and transformed
  capacity feasibility.
- The uncertainty model and sorted-Gamma penalty are stated only for integer
  Gamma in `{0,...,n}`.
- The finite breakpoint result uses the complete original set
  `B = {0} union {|t_ij|}`.  Exact candidates are never tolerance-clustered.
- Fixed-threshold LP geometry states the assumptions needed for baseline
  feasibility, nonnegative costs, segment ordering, zero-cost increments, and
  ties.
- The one-item rounding and additive-gap proof identify the exact feasible
  rounded solution and do not credit optional repair with the theorem.
- The relative `O(1/n)` statement is conditional on uniform jump bounds and a
  linear positive lower bound on the LP value; experiments do not prove it.
- Exact search retains every integer-safe non-dominated original option,
  including points below the LP upper hull.
- A limited global gap is finite only after every original threshold has a
  valid upper-bound record.
- Floating-point incumbents pass the direct sorted-Gamma check.  Any numerical
  feasibility margin and tolerance are recorded; no such check is called
  interval arithmetic.

## Implementation and adversarial gates

- Unit and property tests cover the model reduction, full candidate set,
  fixed-threshold reconstruction, hull LP bound, rounding certificate, direct
  robust certificate, exact search, limited-search bound records, sweep parity,
  and serialization.
- Brute-force enumeration on small random and adversarial instances is the
  reference oracle for global objective/feasibility parity.
- Regression tests retain binary64-distinct close thresholds, including the
  known unique-feasible-breakpoint counterexample.
- Tests cover repeated values, zero deviations, Gamma equal to zero and `n`,
  negative transformed capacities, zero-cost options, duplicate points, ties,
  and below-hull integer optima.
- Public APIs fail closed on malformed, nonfinite, or out-of-domain inputs.

## Evidence protocol

All canonical results are generated into a new dated directory under
`results/release/`; existing raw or dated evidence is immutable.  The protocol
file is committed before the full run and records environment versions,
hardware, seeds, instance grids, stopping rules, exclusions, and planned
aggregations.  The release contains raw rows, generated summaries, a manifest,
and checksums.

The campaign must support only these empirical questions:

1. Does HullRound obey its per-threshold additive certificate?
2. Do exact enumeration and the guarded sweep agree on transformed states,
   incumbents, upper bounds, and completion status?
3. Does exact search agree with brute force on small instances and with open
   solver baselines on matched completed instances?
4. What construction work does the sweep save, without claiming general
   solver-level dominance?
5. How does robust protection trade off against nominal objective in a clearly
   labeled controlled/semi-synthetic pricing study?
6. Where do tight capacity, many thresholds, or time limits produce hard or
   incomplete rows, and what valid gaps remain?

## Manuscript and artifact gates

- Public and anonymous variants are built from one canonical source and one
  generated evidence release.
- Numerical tables, figures, macros, summaries, and manifests are generated,
  not hand-edited.
- The abstract, contributions, related work, theorem statements, algorithm,
  experiment section, limitations, conclusion, README, CITATION.cff, and arXiv
  metadata use the same title and scope.
- Every citation is resolved in the evidence ledger and every cited key is used.
- Both PDFs compile without undefined references, missing files, bad boxes that
  damage readability, or stale claims.
- Every page of public and anonymous PDFs is rendered and visually inspected.
- Anonymous artifacts contain no author, institution, username, repository,
  arXiv identifier, acknowledgments, or identifying PDF metadata.
- The arXiv source package compiles in an isolated directory and contains only
  required source/generated assets.

## Repository and freeze gates

- A fresh clone installs and passes the documented verification path.
- License, citation metadata, contribution boundary, changelog, data policy,
  environment lock/record, artifact manifest, and archival instructions are
  complete.
- The repository is made public only after the release commit passes all gates.
- The Paper A branch is merged, and the exact submitted commit is tagged only
  after the arXiv upload package is final.
- The GitHub release, source archive, public PDF, anonymous PDF, evidence
  checksums, and arXiv source package all name the same commit.
- Paper B is updated only after the frozen Paper A bibliographic record is
  available.

## Stop conditions

The release must not be called final if a theorem has an unresolved edge case,
an empirical number lacks canonical raw evidence, a reference is unresolved,
an anonymous identity scan fails, an isolated build fails, or the repository
snapshot differs from the submitted package.  Failed gates and discovered
counterexamples are recorded rather than weakened or omitted silently.
