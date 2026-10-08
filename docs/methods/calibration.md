# Probability Calibration

A probability forecast is calibrated when events forecast at probability `p`
happen with frequency `p`. The forecast can be a classifier's output, an
analyst's probability, or a prediction-market price read as an implied
probability. Calibration is separate from discrimination: a forecaster that
always predicts the base rate is calibrated but useless, and a model that ranks
events well can still be systematically overconfident.

`ml4t.diagnostic.metrics` provides proper scores, the Brier decomposition, a
reliability table with confidence intervals, calibration-error summaries, and a
calibration slope. All functions take NumPy arrays, Polars or pandas Series,
or sequences of forecasts in `[0, 1]` and outcomes in `{0, 1}`. Missing values
raise an error instead of being dropped.

## Example: prices with a favorite-longshot bias

The synthetic market below quotes prices in cents. The true win probability is
`expit(1.4 * logit(price))`, so longshots win less often than priced and
favorites more often. Markets come in groups of four that resolve on one
event, so outcomes within a group are identical.

```python
import numpy as np

from ml4t.diagnostic.metrics import (
    brier_decomposition,
    brier_score,
    calibration_slope,
    expected_calibration_error,
    log_loss,
    max_calibration_error,
    reliability_table,
)

rng = np.random.default_rng(0)
n_events, markets_per_event = 12_500, 4
event_price = np.clip(np.round(rng.beta(0.6, 0.6, n_events), 2), 0.01, 0.99)
true_prob = 1 / (1 + np.exp(-1.4 * np.log(event_price / (1 - event_price))))
event_outcome = (rng.uniform(size=n_events) < true_prob).astype(int)

price = np.repeat(event_price, markets_per_event)
outcome = np.repeat(event_outcome, markets_per_event)
event_id = np.repeat(np.arange(n_events), markets_per_event)

print(f"Brier score: {brier_score(price, outcome):.4f}")
print(f"Log loss:    {log_loss(price, outcome):.4f}")

decomposition = brier_decomposition(price, outcome)
print(
    f"reliability={decomposition.reliability:.4f} "
    f"resolution={decomposition.resolution:.4f} "
    f"uncertainty={decomposition.uncertainty:.4f}"
)

table = reliability_table(price, outcome, clusters=event_id, n_bootstrap=200, seed=0)
print(table.select("bin_lower", "bin_upper", "n", "n_clusters",
                   "mean_forecast", "observed_frequency", "ci_lower", "ci_upper"))

print(f"ECE: {expected_calibration_error(price, outcome):.4f}")
print(f"MCE: {max_calibration_error(price, outcome):.4f}")

slope = calibration_slope(price, outcome, spec="logit", clusters=event_id)
print(f"logit slope {slope.slope:.2f}, 95% CI {slope.slope_ci[0]:.2f}-{slope.slope_ci[1]:.2f}, "
      f"p(slope=1) = {slope.slope_p_value:.1e}")
```

## Scores

`brier_score` is the mean squared error `(p - y)^2`. A constant forecast of
0.5 scores 0.25. `log_loss` is the mean negative log-likelihood. It punishes
confident errors far more than the Brier score does. Forecasts are clipped to
`[eps, 1 - eps]` (default `1e-15`), because one market priced at exactly 0 or
1 that resolves the other way would otherwise make the loss infinite. Report
how many prices sit at 0 or 1 whenever you quote a log loss.

Both accept `weights`, for example to weight markets by traded volume.

## Brier decomposition

Murphy (1973) splits the Brier score over forecast bins `k`, each with weight
share `W_k / W`, mean forecast `f_k` and observed frequency `o_k`:

```
brier = reliability - resolution + uncertainty
reliability = sum_k (W_k / W) (f_k - o_k)^2     lower is better
resolution  = sum_k (W_k / W) (o_k - o)^2       higher is better
uncertainty = o (1 - o)                         set by the outcomes alone
```

This three-term identity is exact only when all forecasts in a bin are equal.
`brier_decomposition(bins=None)`, the default, uses one bin per distinct
forecast value, which suits prices quoted on a tick grid. With explicit `bins`,
forecasts vary within a bin, and two more terms keep the identity exact
(Stephenson, Coelho and Jolliffe, 2008): `within_bin_variance` and
`within_bin_covariance`, with
`brier = reliability - resolution + uncertainty + within_bin_variance -
within_bin_covariance`. `brier_skill_score` is `1 - brier / uncertainty`.

## Reliability table

`reliability_table` returns one row per bin: `n`, `n_clusters`, the summed
`weight`, `mean_forecast`, `observed_frequency`, `difference` (observed minus
forecast; negative means the bin is overpriced), and an interval for the
observed frequency. Bins are half-open `[lower, upper)`, and the last bin also
holds forecasts equal to 1. Empty bins remain in the table with null
statistics.

The default edges, `PREDICTION_MARKET_BIN_EDGES`, are
`0, .01, .02, .05, .1, .2, ..., .9, .95, .98, .99, 1`. Market prices pile up
near 0 and 1, and the favorite-longshot bias is concentrated there. Equal-width
deciles would put every price below 10 cents into one bin.

The interval method depends on the inputs and is recorded in `ci_method`:

| Inputs | `ci_method` | Interval |
|---|---|---|
| no clusters, no weights | `wilson` | Wilson score interval |
| weights, no clusters | `wilson_effective_n` | Wilson on the Kish effective sample size `(sum w)^2 / sum w^2` |
| `clusters` given | `cluster_bootstrap` | Percentile interval from resampling whole clusters with replacement |

Pass `clusters` whenever outcomes are dependent within a group. Several
markets on one election, or every strike of one price-range contract, resolve
together. A Wilson interval counts them as independent draws and can be too
narrow by a factor of up to `sqrt(markets per event)`. The bootstrap resamples
clusters across all bins at once. Replicates in which a bin is empty are left
out of that bin's interval, so read intervals for bins with few clusters with
care. `n_bootstrap` (default 1000) and `seed` (default 0, so repeated calls
match) control the resampling.

## Calibration error summaries

`expected_calibration_error` is `sum_k (W_k / W) |o_k - f_k|` over non-empty
bins, and `max_calibration_error` is the largest `|o_k - f_k|`. Both depend
on the bin edges, so report the edges with them. Sampling noise alone makes
the expected ECE positive for a perfectly calibrated forecast, and MCE is
often set by the thinnest bin. Check that bin's `n` and interval in the
reliability table.

## Calibration slope and the favorite-longshot bias

`calibration_slope` regresses outcomes on forecasts:

- `spec="linear"`: `outcome = a + b * forecast`, weighted least squares (a
  Mincer-Zarnowitz regression on probabilities).
- `spec="logit"`: `logit P(outcome) = a + b * logit(forecast)`, a logistic
  regression (Cox's calibration slope). Forecasts are clipped to
  `[eps, 1 - eps]` (default `eps=1e-3`) before the transform, because a price
  of 0 or 1 has an infinite logit.

A calibrated forecast has `b = 1` and `a = 0`. **`b > 1` is the
favorite-longshot bias**: outcomes rise faster than prices, so longshots win
less often than priced (overpriced) and favorites more often (underpriced).
`b < 1` is the reverse, the signature of an overconfident forecaster whose
probabilities are too extreme.

Standard errors use the sandwich estimator with the small-sample factor
`G / (G - 1) * (n - 1) / (n - 2)`, where `G` is the number of clusters
(cluster-robust CR1) or `n` without clusters (HC1). Intervals use a t
distribution with `G - 1` degrees of freedom under clustering. `slope_p_value`
tests `b = 1`.

A slope estimate is a summary. A bias confined to a few cents at either end
can leave `b` close to 1, so read the reliability table as well. And a
statistically significant `b` is not yet a trading edge: whether it survives
fees, spreads and capacity limits has to be checked separately.

## References

- Brier, G. W. (1950). "Verification of Forecasts Expressed in Terms of
  Probability." *Monthly Weather Review*, 78(1), 1-3.
- Murphy, A. H. (1973). "A New Vector Partition of the Probability Score."
  *Journal of Applied Meteorology*, 12(4), 595-600.
- Stephenson, D. B., Coelho, C. A. S., & Jolliffe, I. T. (2008). "Two Extra
  Components in the Brier Score Decomposition." *Weather and Forecasting*,
  23(4), 752-757.
- Wilson, E. B. (1927). "Probable Inference, the Law of Succession, and
  Statistical Inference." *Journal of the American Statistical Association*,
  22(158), 209-212.
- Cox, D. R. (1958). "Two Further Applications of a Model for Binary
  Regression." *Biometrika*, 45(3/4), 562-565.
- Mincer, J., & Zarnowitz, V. (1969). "The Evaluation of Economic Forecasts."
  In *Economic Forecasts and Expectations*, NBER, 3-46.
- Cameron, A. C., & Miller, D. L. (2015). "A Practitioner's Guide to
  Cluster-Robust Inference." *Journal of Human Resources*, 50(2), 317-372.
- Snowberg, E., & Wolfers, J. (2010). "Explaining the Favorite-Long Shot Bias:
  Is it Risk-Love or Misperceptions?" *Journal of Political Economy*, 118(4),
  723-746.
