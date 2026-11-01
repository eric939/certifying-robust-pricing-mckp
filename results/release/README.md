# Evidence releases

Each dated folder is immutable once released: it is never rewritten, and every
file is covered by its manifest. Newer releases are added beside older ones.

| Release | Paper version | Produced by | Verify with |
|---|---|---|---|
| `2026-10-31-paper-a-revision-protocol-v1` | revised (§7, protocol v1: Panels S, L; 1,764 runs) | `studies/paper-a-revision` | `make verify-evidence` |
| `2026-10-31-paper-a-revision-protocol-v2` | revised (§7, protocol v2: interval records, SCIP, Panel H, price of robustness; 1,260 runs) | `studies/paper-a-revision` | `make verify-evidence` |
| `2026-08-29-paper-a-v3-independent-rq` | submitted (separated reference price and unit cost) | tag `v3.1.1` | `make verify-archive` |
| `2026-08-28-paper-a-v3` | submitted (certificate study, pilot) | tag `v3.1.1` | `make verify-archive` |
| `2026-08-09-paper-a-final-r2` | submitted (§7 of the submitted version) | tag `v3.1.1` | `make verify-archive` |

In the two 2026-10-31 releases the public tree omits files that are
regenerated and checked against the manifest by the verifier: the protocol-v1
base portfolios (`instances/`, 139 MB, from their seeds) and the generated
LaTeX tables and figure (`reporting/*.tex`, `reporting/*.pdf`, from the raw
records). The complete folders, including these files, are Online Resource 3
of the revised paper.
