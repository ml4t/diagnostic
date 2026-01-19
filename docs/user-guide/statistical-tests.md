# Statistical Tests

ML4T Diagnostic implements rigorous statistical tests to prevent false discoveries and account for multiple testing bias.

## Deflated Sharpe Ratio (DSR)

The DSR adjusts the Sharpe ratio for the number of backtests tried:

```python
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

result = deflated_sharpe_ratio(
    returns=strategy_returns,
    num_trials=100,           # How many strategies tested
    frequency='daily',        # Return frequency
    annualization_factor=252
)

print(f"Observed Sharpe: {result.sharpe:.2f}")
print(f"Deflated Sharpe: {result.dsr:.2f}")
print(f"p-value: {result.pvalue:.4f}")
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
from ml4t.diagnostic.evaluation.stats import compute_ras

result = compute_ras(
    returns=strategy_returns,
    n_bootstraps=1000,
    kappa=0.0  # 0 = pure complexity, 1 = Sharpe only
)

print(f"RAS-adjusted Sharpe: {result.ras_sharpe:.2f}")
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
from ml4t.diagnostic.evaluation.stats import fdr_correction

adjusted_pvalues = fdr_correction(
    pvalues=[0.01, 0.03, 0.05, 0.08, 0.12],
    method='bh'  # Benjamini-Hochberg
)

# Identify discoveries
discoveries = adjusted_pvalues < 0.05
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
from ml4t.diagnostic.evaluation.stats import compute_hac_stats

result = compute_hac_stats(
    ic_series=information_coefficients,
    bandwidth='auto'  # Automatic bandwidth selection
)

print(f"HAC t-stat: {result.tstat:.2f}")
print(f"HAC std error: {result.std_error:.4f}")
```

## Probability of Backtest Overfitting (PBO)

Estimate the probability that an optimal strategy is overfit:

```python
from ml4t.diagnostic.evaluation.stats import probability_backtest_overfitting

pbo = probability_backtest_overfitting(
    strategy_returns_matrix,  # Returns from all tried strategies
    n_partitions=16
)

print(f"PBO: {pbo:.1%}")  # e.g., "32.5%"
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
