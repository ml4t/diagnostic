# Contributing to ml4t-diagnostic

Follow the [ML4T contribution policy](https://github.com/ml4t/ecosystem/blob/main/CONTRIBUTING.md)
for issue scope, pull requests, regression evidence, and AI-assistance disclosure. This repository
owns changes to `ml4t.diagnostic`; shared policy and qualification changes belong in
`ml4t/ecosystem`.

Search [existing issues](https://github.com/ml4t/diagnostic/issues) before filing a report. Use
the repository's bug, feature, or documentation form for one problem at a time. Report suspected
vulnerabilities through [private security reporting](SECURITY.md), not an issue.

Each user-visible change or defect fix needs an owning issue and one focused pull request.
Automated dependency updates and administrative metadata changes do not need a separate issue.
For a bug fix, run a regression through the affected public API on the base and proposed
revisions, and report both results. Correct verified numerical defects. Before changing an
intended metric definition, annualization, sign, or default, agree on expected behavior and
compatibility impact in the owning issue.

The [repository guide](AGENTS.md) lists the quality commands. Include compatibility,
documentation, release impact, and AI assistance in the pull request. Review and test any
AI-assisted submission yourself, and be able to explain it to reviewers.
