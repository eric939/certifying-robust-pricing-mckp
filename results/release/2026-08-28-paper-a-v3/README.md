# Paper A v3 evidence bundle

This directory contains the evidence behind the numerical claims in the v3
manuscript. It was assembled on 28 August 2026. Provider-controlled raw
Dominick's files are not redistributed.

## Controlled certificate study

`global-certificate-study/` contains 45 shared full-family upper certificates
paired with three incumbent tracks, serialized as 135 exact JSON intervals:
literal lower-endpoint rounding, rounding/repair plus completion, and the
tolerance-controlled branch-and-bound incumbent. `protocol.json` records
the base commit, dirty-tree state, input release hash, source hashes, and exact
arithmetic policy. `MANIFEST.sha256` covers the study archive.

The reported relative width is `(U-H)/U` only when `U > 0`. Every stored
multiplier passes exact left/right subgradient optimality, so `U` is the sharp
full-family LP endpoint for the encoded dyadic-rational instance.

`global-certificate-study/instances/` stores all 45 controlled inputs with
every binary64 coefficient encoded by `float.hex()`. These exact snapshots make
verification portable: the original transcendental generator calculations can
differ by one final bit across numerical platforms.

## Dominick's pilot

`dominicks-pilot/dominicks_cereal_pilot.json` contains the complete derived
pilot record. Its three feasible robust targets have exact certificates under
`dominicks-pilot/certificates/`. `RAW_DATA_PROVENANCE.md` records the official
source, provider terms, manual caveats, and SHA-256 hashes of the two locally
obtained source files. The UPC dictionary is provenance-only and is not read
by the computation.

The pilot is a post-hoc predictive semisynthetic common-shock transport stress
test. It is not a causal, off-policy, preregistered, or untouched evaluation.
The training-frozen roster is fully
observed at a positive price in only 7 of 79 future weeks, so the availability
gate fails and no credible portfolio-level validation is claimed. Later-block
shocks are applied to the fixed week-260 demand vector encoded in the pricing
instance rather than to week-specific candidate-price forecasts.

## Source snapshot

`source-snapshot/` records the v3-specific source state that generated the
original evidence, layered on top of the base commit recorded in the
controlled-study protocol. It preserves that provenance and does not replace
the current public repository source.

The manuscript is maintained separately and is not included here. The
top-level `MANIFEST.sha256` binds the controlled archive, exact input snapshots,
pilot outputs, and evidence-generating source snapshot. This reproducibility
bundle is published with repository tag `v3.0.1`.
