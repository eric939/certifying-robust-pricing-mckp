# v3-specific code snapshot

Apply these paths over commit
`73e8ca28e1c1c237cafea26cff7c24fe25489b02` of the
`certifying-robust-pricing-mckp` repository.

Included files:

- `src/robust_mckp/global_certificate.py`: exact global certificate builder and
  verifier.
- `src/robust_mckp/__init__.py`: public exports for the certificate API.
- `tests/test_global_certificate.py`: deterministic, randomized, and
  adversarial certificate checks.
- `tests/test_v3_dominicks_pilot.py`: regression checks that prevent
  future-conditioned roster filling and verify the recorded availability
  failure.
- `scripts/run_v3_global_certificate_study.py`: frozen-panel study and archive
  builder.
- `scripts/run_v3_dominicks_pilot.py`: chronological provider-data pilot and
  application-certificate builder.
- `scripts/plot_v3_dominicks_diagnostics.py`: availability and score diagnostic
  figure builder.

The full repository test suite passed 122 tests in the recorded development
environment. The builder and verifier remain in the same module and share
exact-algebra helpers; the manuscript discloses that boundary rather than
claiming a separately implemented checker.
