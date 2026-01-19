# Trade Analysis

Analyze individual trade performance and identify systematic error patterns using SHAP.

## Basic Trade Analysis

```python
from ml4t.diagnostic.evaluation import TradeAnalysis

analyzer = TradeAnalysis(trade_records)

# Summary statistics
print(analyzer.summary())

# Identify worst trades
worst = analyzer.worst_trades(n=20)
best = analyzer.best_trades(n=20)
```

### Trade Metrics

| Metric | Description |
|--------|-------------|
| Win Rate | % of profitable trades |
| Profit Factor | Gross profit / gross loss |
| Average Win/Loss | Mean P&L by outcome |
| Sharpe Ratio | Risk-adjusted return |
| Max Drawdown | Largest peak-to-trough |

## SHAP-Based Error Analysis

Identify why trades fail using SHAP explanations:

```python
from ml4t.diagnostic.evaluation import TradeShapAnalyzer

shap_analyzer = TradeShapAnalyzer(
    model=trained_model,
    features_df=features_df,
    shap_values=precomputed_shap
)

# Explain worst trades
result = shap_analyzer.explain_worst_trades(worst_trades)

# View error patterns
for pattern in result.error_patterns:
    print(f"\nCluster {pattern.cluster_id} ({pattern.n_trades} trades)")
    print(f"  Top features: {pattern.top_features}")
    print(f"  Hypothesis: {pattern.hypothesis}")
    print(f"  Actions: {pattern.actions}")
```

### Error Pattern Discovery

The analyzer automatically:

1. **Clusters** failed trades by SHAP similarity
2. **Identifies** common feature patterns in each cluster
3. **Generates** hypotheses about failure causes
4. **Suggests** actionable improvements

## Example Output

```
Cluster 0 (12 trades):
  Top features: ['volatility_20d', 'momentum_fast']
  Hypothesis: "Trades fail during high volatility regimes
               when fast momentum gives false signals"
  Actions: ["Add volatility filter", "Increase momentum lookback"]

Cluster 1 (8 trades):
  Top features: ['sector_tech', 'earnings_surprise']
  Hypothesis: "Tech sector trades around earnings are unpredictable"
  Actions: ["Reduce position size around earnings",
            "Add earnings calendar filter"]
```

## Trade Filtering

Focus analysis on specific trade types:

```python
# Filter by symbol
tech_trades = analyzer.filter(sector='Technology')

# Filter by time period
q4_trades = analyzer.filter(
    start_date='2025-10-01',
    end_date='2025-12-31'
)

# Filter by outcome
losers = analyzer.filter(pnl_lt=0)
```

## Attribution Analysis

Break down performance by dimension:

```python
# By sector
sector_attr = analyzer.attribute_by('sector')

# By time period
monthly_attr = analyzer.attribute_by_period('monthly')

# By market regime
regime_attr = analyzer.attribute_by('regime')
```

## Excursion Analysis

Optimize stop-loss and take-profit levels:

```python
from ml4t.diagnostic.evaluation import analyze_excursions

result = analyze_excursions(
    trade_records,
    price_data,
    tp_levels=[0.01, 0.02, 0.03],
    sl_levels=[0.01, 0.02, 0.03]
)

# Optimal parameters
print(f"Optimal TP: {result.optimal_tp:.2%}")
print(f"Optimal SL: {result.optimal_sl:.2%}")
```

## Visualization

```python
# Trade P&L distribution
analyzer.plot_pnl_distribution()

# Cumulative P&L
analyzer.plot_equity_curve()

# SHAP summary for worst trades
shap_analyzer.plot_shap_summary(worst_trades)

# Error cluster visualization
result.plot_clusters()
```

## Export Reports

```python
# HTML report
analyzer.to_html("trade_analysis.html")

# JSON for further analysis
analyzer.to_json("trade_analysis.json")
```
