## Owning issue

Closes #

Automated dependency updates and administrative metadata changes do not need a separate issue.
Each other PR addresses one issue and contains one focused change. See the
[contribution policy](https://github.com/ml4t/diagnostic/blob/main/CONTRIBUTING.md).

## Outcome

Describe the user-visible or standards outcome.

## Compatibility

- [ ] No public compatibility impact
- [ ] Compatibility impact is documented in the owning issue and migration guidance

## Verification

- [ ] For a bug: a public-API or user-workflow regression fails on the base revision and passes on this head
- [ ] Ruff lint and format checks pass
- [ ] `ty` passes
- [ ] Test suite passes
- [ ] Package build passes when applicable
- [ ] Strict MkDocs build passes when documentation is affected
- [ ] Ecosystem qualification passes

For a bug fix, record the base and head commit IDs, the regression command, and each result.
For other changes, report the checks actually run and any missing evidence. A checked box is not
a substitute for results.

## AI assistance

State whether AI tools helped produce code, tests, documentation, or this description. Name the
parts they helped with. Confirm that you reviewed and tested the submission yourself and can
explain it to reviewers.

## Documentation and release

State the documentation, release notes, and patch-release work required after merge.
