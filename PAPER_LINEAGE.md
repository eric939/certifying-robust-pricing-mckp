# Paper lineage and contribution boundary

This repository is **Paper A**, the continuation of
arXiv:2603.18653, *Robust Discrete Pricing Optimization via Multiple-Choice
Knapsack Reductions*. Its next public manuscript is intended to replace that
record as version 2 after the source, evidence, and release gates are complete.

The separate repository `eric939/robust_mckp` is **Paper B**, *Simultaneous
Group-Envelope Bounds for Gamma-Robust Multiple-Choice Knapsack Problems*. It
will receive a new arXiv identifier.

## Paper A owns

- the finite-menu pricing model and exact MCKP reduction;
- the integer-budget robust margin formulation;
- the complete original deviation-breakpoint decomposition;
- fixed-threshold MCKP hull geometry and LP bounds;
- HullRound and the one-item additive rounding certificate;
- exact full-breakpoint integer search and valid global-gap accounting;
- the guarded parametric threshold sweep; and
- pricing-oriented controlled computational evidence.

## Paper B owns

- the exactly-one baseline cancellation across threshold intervals;
- simultaneous group-envelope evaluation over all retained thresholds;
- the interval minimax bound and dominance over the group-clique relaxation;
- certified multiplier bracketing and minimization;
- adaptive LP-family certification using the interval bound; and
- the matched interval, scaling, external-coefficient, and integer-integration
  evidence released with that algorithm.

## Shared material

Paper B may restate enough of the robust MCKP model, Bertsimas--Sim threshold
reduction, fixed-threshold relaxation, and pricing specialization to be
self-contained. Those elements must be cited to Paper A and described as
background rather than as new Paper B contributions. Reused prose should be
limited and the relationship should be disclosed in the arXiv Comments field.

Paper A must not import Paper B's simultaneous envelope, minimax-dominance,
certified multiplier-search, or Paper B computational results. Paper A should
end by identifying simultaneous all-threshold bounding as later work, not as a
result of Paper A.

## Release order

1. Finish, verify, and freeze Paper A's arXiv-v2 source and evidence.
2. Submit Paper A as arXiv:2603.18653v2 and record its exact public version.
3. Update Paper B's bibliography and relationship paragraph to the frozen
   Paper A metadata.
4. Submit Paper B as a new arXiv paper with a new identifier.

Neither paper supersedes the other. Paper B builds on Paper A's model and
fixed-threshold machinery while answering a different algorithmic question.
