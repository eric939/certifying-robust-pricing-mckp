# Paper A closest-work search log

Search date: 2026-08-09 (Europe/Zurich)  
Purpose: support the qualified positioning claim A-L3, not an exhaustive
systematic-review or universal priority claim.

## Scope and screening rule

The search looked for work at the intersection of (1) finite-menu or
constrained product pricing, (2) multiple-choice knapsack structure, (3)
budgeted/Gamma robustness or finite deviation thresholds, and (4) computable
LP, rounding, or exact-search certificates. A result was retained when its
title, abstract, or primary-source description covered at least two of those
dimensions, or when it was a foundational source for an ingredient explicitly
used by Paper A. Citation chasing from retained primary sources was used for
classical MCKP and robust-knapsack foundations.

Metadata and semantic claims were accepted only after resolution to a primary
publisher page, journal registry, or the arXiv record. The exact locators and
semantic support are recorded in `EVIDENCE_LEDGER_A.csv`. Search-engine and API
results were discovery aids, not evidence by themselves.

## Reproducible registry searches

The Crossref REST endpoint was queried with `query.bibliographic`, sorted by
Crossref relevance, and the first ten records were screened (five records for
queries 1, 3, and 4 in the initial pass; query 2 was rerun with ten after a
rate-limit response):

1. `robust discrete pricing multiple-choice knapsack budgeted uncertainty`
2. `robust multiple-choice knapsack budgeted uncertainty`
3. `portfolio pricing ratio margin robust optimization`
4. `discrete pricing knapsack margin fairness`

Reproduction template:

```text
GET https://api.crossref.org/works
    ?query.bibliographic=<URL-ENCODED QUERY>
    &rows=10
    &select=DOI,title,author,published,container-title,type
```

Relevant retained hits included Caserta and Voss
(`10.1016/j.omega.2018.06.014`), the MCKP chapter in Kellerer, Pferschy, and
Pisinger (`10.1007/978-3-540-24777-7_11`), and adjacent robust combinatorial
optimization records. The other top records were screened out because they
concerned unrelated scheduling, bin packing, financial-asset pricing,
interdiction, or non-pricing knapsack variants.

The arXiv API was queried on the same date with `start=0&max_results=10`:

| API query | Results screened | Relevant result |
|---|---:|---|
| `all:"robust discrete pricing" AND all:knapsack` | 1 | Paper A v1 only |
| `all:"multiple-choice knapsack" AND all:"budgeted uncertainty"` | 0 | none |
| `all:"multiple choice knapsack" AND all:"budgeted uncertainty"` | 0 | none |
| `all:"ratio margin" AND all:pricing AND all:robust` | 0 | none |

Reproduction template:

```text
GET https://export.arxiv.org/api/query
    ?search_query=<URL-ENCODED QUERY>
    &start=0
    &max_results=10
```

## Broader discovery queries

The following exact phrases were also run through a general scholarly web
search to catch publisher pages whose metadata vocabulary differs from the
paper's terminology:

- `robust discrete pricing multiple-choice knapsack budgeted uncertainty`
- `robust multiple-choice knapsack budgeted uncertainty scalar threshold`
- `portfolio pricing ratio margin robust MCKP`
- `fixed threshold MCKP robust pricing branch and bound certificate`

This pass rediscovered Paper A, Caserta and Voss, Bertsimas and Sim, and robust
single-knapsack work. Citation chasing and terminology expansion added the
publisher records for Thiele; Hamzeei, Lim, and Xu; Harsha, Subramanian, and
Ettl; Shao, Mai, and Cheng; Monaci, Pferschy, and Serafini; Buesing, Gersing,
and Koster; and the classical MCKP sources listed in the evidence ledger.

## Audit conclusion and limitation

The retained sources cover every major ingredient separately, and several
cover important pairs of ingredients. In particular, scalar budgeted-robust
reformulation, finite deviation families, MCKP LP geometry, robust knapsack,
constrained discrete pricing, and robust multiproduct pricing all predate Paper
A. The reviewed primary sources did not state the same conjunction of the
ratio-margin finite-menu pricing model, complete original breakpoint family,
fixed-threshold one-item certificate, exact full-family gap accounting, and
direct failure-safe implementation. That scoped comparison supports Paper A's
qualified composition claim in `CLOSEST_WORK_COMPARISON.md`; it does not prove
that no uncatalogued work exists and is not used for an exhaustive negative
novelty statement.
