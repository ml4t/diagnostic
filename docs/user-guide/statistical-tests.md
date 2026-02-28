# Statistical Tests

ML4T Diagnostic implements rigorous statistical tests to prevent false discoveries and account for multiple testing bias.

## Deflated Sharpe Ratio (DSR)

The DSR adjusts the Sharpe ratio for the number of backtests tried:

```python
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

result = deflated_sharpe_ratio(
    returns=strategy_returns,
    n_trials=100,             # How many strategies tested
    frequency='daily',        # Return frequency
    periods_per_year=252
)

print(f"Observed Sharpe: {result.sharpe_ratio:.2f}")
print(f"Deflated Sharpe: {result.deflated_sharpe:.2f}")
print(f"p-value: {result.p_value:.4f}")
```

### When to Use DSR

- After trying multiple strategy variations
- When selecting among several candidate strategies
- To report statistically honest performance

### DSR Formula (López de Prado et al. 2025)

$$DSR = \Phi^{-1}\left(1 - e^{-\frac{1}{2}\gamma}\right)$$

where $\gamma$ accounts for:
- Number of trials
- Expected maximum Sharpe under null hypothesis
- Autocorrelation in returns

See [Deflated Sharpe Ratio Guide](../guides/deflated-sharpe-ratio.md) for details.

## Rademacher Anti-Serum (RAS)

RAS detects backtest overfitting using complexity theory:

```python
from ml4t.diagnostic.evaluation.stats import rademacher_complexity, ras_sharpe_adjustment

# returns_matrix shape: (n_periods, n_strategies)
complexity = rademacher_complexity(returns_matrix)
observed_sharpes = returns_matrix.mean(axis=0) / returns_matrix.std(axis=0)
result = ras_sharpe_adjustment(
    observed_sharpe=observed_sharpes,
    complexity=complexity,
    n_samples=returns_matrix.shape[0],
    n_strategies=returns_matrix.shape[1],
    return_result=True,
)

print(f"Number significant after RAS: {result.n_significant}")
print(f"Complexity penalty: {result.complexity:.4f}")
```

### Interpretation

| RAS Result | Interpretation |
|------------|----------------|
| High RAS | Strategy is robust, not overfit |
| Low RAS | Strategy may be overfit |
| Negative RAS | Strategy is likely spurious |

## Minimum Track Record Length (MinTRL)

Calculate how long a track record must be for statistical significance:

```python
from ml4t.diagnostic.evaluation.stats import compute_min_trl

result = compute_min_trl(
    sharpe_ratio=1.5,
    target_pvalue=0.05,
    frequency='daily'
)

print(f"Minimum observations: {result.min_observations}")
print(f"Minimum years: {result.min_years:.1f}")
```

### MinTRL with Multiple Testing

For FWER-controlled significance across multiple strategies:

```python
from ml4t.diagnostic.evaluation.stats import min_trl_fwer

result = min_trl_fwer(
    sharpe_ratio=1.5,
    num_trials=50,
    alpha=0.05
)
```

## False Discovery Rate (FDR)

Control the expected proportion of false positives:

```python
from ml4t.diagnostic.evaluation.stats import benjamini_hochberg_fdr

pvalues = [0.01, 0.03, 0.05, 0.08, 0.12]
rejected = benjamini_hochberg_fdr(p_values=pvalues, alpha=0.05)

# Identify discoveries
discoveries = rejected
```

### Methods

| Method | Description |
|--------|-------------|
| `bh` | Benjamini-Hochberg (controls FDR) |
| `by` | Benjamini-Yekutieli (conservative) |
| `holm` | Holm-Bonferroni (controls FWER) |

## HAC-Adjusted Statistics

Account for heteroskedasticity and autocorrelation:

```python
from ml4t.diagnostic.evaluation.stats import hac_adjusted_ic

result = hac_adjusted_ic(
    predictions=predictions,
    returns=forward_returns,
    return_details=True,
)

print(f"HAC t-stat: {result['t_stat']:.2f}")
print(f"HAC std error: {result['bootstrap_std']:.4f}")
```

## Probability of Backtest Overfitting (PBO)

Estimate the probability that an optimal strategy is overfit:

```python
from ml4t.diagnostic.evaluation.stats import compute_pbo

result = compute_pbo(
    is_performance=is_returns_matrix,
    oos_performance=oos_returns_matrix,
)

print(f"PBO: {result.pbo:.1%}")  # e.g., "32.5%"
```

### Interpretation

| PBO | Interpretation |
|-----|----------------|
| < 10% | Low overfitting risk |
| 10-30% | Moderate risk |
| > 30% | High overfitting risk |

## References

- López de Prado et al. (2025). "How to Use the Sharpe Ratio"
- Bailey & López de Prado (2014). "The Deflated Sharpe Ratio"
- Paleologo, G. (2024). *Elements of Quantitative Investing*
