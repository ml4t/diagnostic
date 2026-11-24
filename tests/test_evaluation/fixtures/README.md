# Sharpe Ratio Inference Fixtures

`dsr_reference.py` holds the reference values for
`tests/test_evaluation/test_dsr_validation.py`. Each case names where its
expected value comes from.

## Source

López de Prado, M., Lipton, A., & Zoonekynd, V., "How to Use the Sharpe Ratio",
ADIA Lab Research Paper Series, No. 19, version of 7 March 2026. Equation and
exhibit numbers in the fixtures refer to that version.

Earlier versions of the paper present the worked example without serial
correlation and print different values for it (PSR 0.987 instead of 0.966, DSR
0.416 instead of 0.410).

## Contents

| Name | Cases | Origin of the expected values |
|---|---|---|
| `PSR_CASES` | 7 | 2 published (0.966, 0.900); 5 computed from Equations 5 and 9 |
| `MINTRL_CASES` | 6 | 2 published (19.543, 39.369 months); 4 computed from Equation 11 |
| `DSR_CASES` | 4 | 1 published (0.410); 3 computed from Equations 28 and 30 |
| `VARIANCE_RESCALING_FACTORS` | 100 | Exhibit 4, K = 1 to 100 |

"Computed" values were calculated from the cited equations outside the library.

## Conventions

- Sharpe ratios are not annualized.
- Kurtosis is Pearson kurtosis (3 for normal returns). The library takes excess
  kurtosis, so the tests subtract 3.
- The paper prints the Sharpe ratio of its example as 0.456. The fixtures use
  the unrounded `PAPER_SHARPE = 0.036 / 0.079`; with 0.456 the DSR is 0.4106,
  which rounds to 0.411 and not to the published 0.410.
- `expected_mintrl` is the value of Equation 11. `compute_min_trl` adds one
  observation and rounds up, so the tests expect `ceil(1 + expected_mintrl)`.

## Not covered

The paper's example also reports a false negative rate (0.411) and two false
discovery rates (0.433 and 0.361). The library has no function for these, so
no test covers them.
