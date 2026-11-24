# Rademacher Anti-Serum validation

The maintained mathematical assumptions and runnable IC example are in
[the statistical references](../../../docs/reference/references.md#rademacher-anti-serum-ras).
This file describes the tests; it does not reproduce unverified published simulation tables.

Run the RAS tests from the repository root:

```bash
uv run pytest tests/test_evaluation/test_ras_ic_population_bound.py tests/test_evaluation/test_ras_validation.py
```

`test_ras_ic_population_bound.py` checks the public cross-sectional IC, adjustment,
configuration and plotting routes. It includes:

- Exact binomial coverage over every count of a bounded two-point IC population.
  The old mean-scale default misses the true mean in 22.8% of samples at `delta=0.05`.
- A rare `-1` IC outcome which a finite sample misses with probability 7.94%.
  The observed sample maximum cannot certify a population bound.
- Period violations hidden by cancelling means, sample/mean mismatches and invalid inputs.
- Valid externally justified bounds, unchanged positional argument/result contracts,
  retained negative bounds, and the actual penalty displayed by the chart.

A singleton strategy has exact empirical Rademacher complexity zero, so these coverage
checks isolate the estimation error without Monte Carlo uncertainty. Multi-strategy
coverage checks use an exact sign expectation rather than a simulated point estimate.

`test_ras_validation.py` also exercises complexity, Sharpe and IC adjustment components.
Its finite simulation outcomes are regression checks, not a proof of zero false discoveries.
IID period vectors, a fixed strategy class, an almost-sure population bound and an exact
or conservative empirical complexity are required for the IC confidence statement.
Ordinary Monte Carlo estimates do not include a confidence allowance for simulation error.
