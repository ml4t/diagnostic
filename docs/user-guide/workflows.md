# Workflows

Practical recipes for the five most common ml4t-diagnostic workflows.
Each section starts with "I have **X**, I want **Y**" so you can jump
straight to what you need.

!!! info "See it in the book"
    Use the [Book Guide](../book-guide/index.md) for exact notebook paths. The shortest
    mapping is: Ch06 for cross-validation, Ch07 for multiple testing, Ch08-Ch09 for
    feature and signal diagnostics, Ch16-Ch17 for backtest and portfolio workflows, and
    Ch19 for trade-level diagnostics and risk analysis.

---

## 1. Signal Analysis

> **I have** a factor DataFrame and price data.
> **I want** IC statistics, quantile spreads, and a tear sheet.

### Quick path (one function)

```python
import polars as pl
from ml4t.diagnostic import analyze_signal

# factor: date | asset | factor  (higher = predicted higher return)
# prices: date | asset | price
factor = pl.read_parquet("momentum_factor.parquet")
prices = pl.read_parquet("prices.parquet")

result = analyze_signal(
    factor,
    prices,
    periods=(1, 5, 21),   # forward return horizons in trading days
    quantiles=5,
    ic_method="spearman",
)

result.summary()
# Prints IC, t-stat, monotonicity, spread for each period

result.to_json("momentum_signal.json")  # save for later
```

### Visualize

```python
from ml4t.diagnostic.visualization.signal import (
    plot_ic_ts,
    plot_quantile_returns_bar,
    plot_cumulative_returns,
    SignalDashboard,
)

# Individual plots
fig = plot_ic_ts(result, period="5D", theme="default")
fig.show()

fig = plot_quantile_returns_bar(result.quantile_result, period="5D")
fig.show()

# Full HTML tear sheet
dashboard = SignalDashboard(title="Momentum Factor")
dashboard.save("momentum.html", result)
```

### HAC-adjusted IC (for autocorrelated signals)

Standard IC t-statistics overstate significance when the IC series is
autocorrelated. Use HAC (Heteroskedasticity and Autocorrelation Consistent)
standard errors instead:

```python
from ml4t.diagnostic.evaluation.metrics import compute_ic_hac_stats

hac = compute_ic_hac_stats(
    ic_series_df,   # DataFrame with "ic" column
    ic_col="ic",
    kernel="bartlett",
)
print(f"HAC t-stat: {hac['t_stat_hac']:.2f} (p={hac['p_value_hac']:.4f})")
```

### Multi-signal comparison (with FDR correction)

When evaluating many candidate signals, correct for multiple testing:

```python
from ml4t.diagnostic.evaluation import MultiSignalAnalysis

signals = {
    "momentum_12m": mom_df,
    "value_ep": val_df,
    "quality_roe": qual_df,
    # ... up to hundreds of signals
}

analyzer = MultiSignalAnalysis(signals, prices)
summary = analyzer.compute_summary()

print(f"Signals tested: {summary.n_signals}")
print(f"FDR-significant: {summary.n_fdr_significant}")
print(f"Top signal: {summary.signal_rankings[0]}")
```

---

## 2. Portfolio Analysis

> **I have** a time series of daily returns (and optionally a benchmark).
> **I want** Sharpe, drawdown analysis, rolling metrics, and a dashboard.

If you need the newer multi-surface reporting flow for backtests, including
`BacktestResult` and artifact-driven tearsheets, see
[Backtest Tearsheets](backtest-tearsheets.md).

### Quick path

```python
import numpy as np
from ml4t.diagnostic.evaluation import PortfolioAnalysis

returns = np.load("strategy_returns.npy")      # daily, non-cumulative
benchmark = np.load("spy_returns.npy")          # optional

analysis = PortfolioAnalysis(
    returns,
    benchmark=benchmark,
    risk_free=0.05,         # annual risk-free rate
    periods_per_year=252,
)

# Summary statistics
metrics = analysis.compute_summary_stats()
print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
print(f"Max DD: {metrics.max_drawdown:.1%}")
print(f"Calmar: {metrics.calmar_ratio:.2f}")

# Drawdown analysis
dd = analysis.compute_drawdown_analysis(top_n=5)

# Monthly returns matrix (for heatmap)
monthly = analysis.get_monthly_returns_matrix()
```

### All-in-one tear sheet

```python
tear_sheet = analysis.create_tear_sheet(theme="default")
tear_sheet.show()                     # Jupyter or browser
tear_sheet.save_html("report.html")   # standalone HTML
```

### Individual plots

```python
from ml4t.diagnostic.visualization.portfolio import (
    plot_cumulative_returns,
    plot_monthly_returns_heatmap,
    plot_drawdown_underwater,
    plot_rolling_sharpe,
)

fig = plot_cumulative_returns(analysis, theme="dark")
fig.show()

fig = plot_monthly_returns_heatmap(monthly, theme="print")
fig.write_image("monthly_heatmap.pdf")  # requires kaleido
```

### Standalone metric functions

If you only need a specific metric and don't want the full analysis object:

```python
from ml4t.diagnostic.evaluation.portfolio_analysis import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    value_at_risk,
)

sr = sharpe_ratio(returns, risk_free=0.05)
md = max_drawdown(returns)
var = value_at_risk(returns, level=0.05)
```

---

## 3. Feature Importance

> **I have** a fitted model, feature matrix `X`, and targets `y`.
> **I want** MDI + permutation + SHAP importance with consensus ranking.

### Quick path

```python
from ml4t.diagnostic.evaluation import analyze_ml_importance

result = analyze_ml_importance(
    model,                         # fitted sklearn-compatible model
    X,                             # DataFrame or array
    y,                             # targets
    methods=["mdi", "pfi", "shap"],
    n_repeats=10,                  # permutation importance repeats
)

# Consensus ranking (Borda count across methods)
print("Top features:", result["consensus_ranking"][:10])

# Agreement between methods
print("MDI vs SHAP:", result["method_agreement"]["mdi_vs_shap"])

# Auto-generated interpretation
print(result["interpretation"])
```

### Visualize

```python
from ml4t.diagnostic.visualization import (
    plot_importance_bar,
    plot_importance_heatmap,
    generate_combined_report,
    export_figures_to_pdf,
)

# Bar chart of top features
fig = plot_importance_bar(result)
fig.show()

# Method comparison heatmap
fig = plot_importance_heatmap(result)
fig.show()

# Full HTML report
generate_combined_report(result, "importance_report.html")

# PDF for publication
export_figures_to_pdf([fig], "importance.pdf")
```

### Feature interactions (SHAP pairwise)

```python
from ml4t.diagnostic.evaluation import compute_shap_interactions

interactions = compute_shap_interactions(
    model, X[:200],                # subsample for speed
    feature_names=X.columns,
    max_samples=200,
)

print("Top interactions:", interactions["top_interactions"][:5])
```

---

## 4. Trade Error Analysis (Trade-SHAP)

> **I have** a list of trades and a fitted model.
> **I want** SHAP explanations for losing trades and clustered error patterns.

### Quick path

```python
from ml4t.diagnostic.evaluation import TradeShapAnalyzer, TradeAnalysis
from ml4t.diagnostic.integration import TradeRecord

# Build trade records (from backtest or manual)
trades = [
    TradeRecord(
        timestamp=exit_dt,
        symbol="AAPL",
        entry_price=150.0,
        exit_price=145.0,
        pnl=-500.0,
        duration=timedelta(days=3),
    ),
    # ...
]

# Rank and select worst trades
ta = TradeAnalysis(trades)
worst = ta.worst_trades(n=20)

# Explain with SHAP
analyzer = TradeShapAnalyzer(
    model,                         # fitted model
    features_df,                   # must have 'timestamp' column + feature columns
)

result = analyzer.explain_worst_trades(worst, n=20)
```

### Interpret results

```python
# Clustered error patterns
for pattern in result.error_patterns:
    print(f"Cluster {pattern.cluster_id}: {pattern.n_trades} trades")
    print(f"  {pattern.description}")
    print(f"  Hypothesis: {pattern.hypothesis}")
    print(f"  Confidence: {pattern.confidence:.0%}")
    for action in pattern.actions:
        print(f"  -> {action}")
```

### Interactive dashboard

```python
# Launch Streamlit dashboard (requires pip install ml4t-diagnostic[dashboard])
# streamlit run -m ml4t.diagnostic.evaluation.trade_shap_dashboard
```

---

## 5. Cross-Validation

> **I have** time-series data with leakage concerns.
> **I want** purged walk-forward or combinatorial CV with proper embargo.

### Walk-forward CV

```python
from ml4t.diagnostic.splitters import WalkForwardCV

cv = WalkForwardCV(
    n_splits=5,
    test_size="4W",        # 20 trading sessions (not 28 calendar days)
    gap=0,
    embargo_pct=0.01,      # 1% embargo after each test fold
    expanding=True,         # expanding training window
    calendar="NYSE",        # calendar-aware session counting
)

scores = []
for train_idx, test_idx in cv.split(X):
    model.fit(X[train_idx], y[train_idx])
    scores.append(model.score(X[test_idx], y[test_idx]))
```

### Combinatorial Purged CV (CPCV)

CPCV generates C(N, k) train/test splits from N contiguous groups,
providing far more paths than standard walk-forward:

```python
from ml4t.diagnostic.splitters import CombinatorialCV

cv = CombinatorialCV(
    n_groups=10,           # 10 contiguous time blocks
    n_test_groups=2,       # 2 blocks held out per split
    embargo_pct=0.01,
    label_horizon=21,      # purge 21 bars of label leakage
)

# C(10, 2) = 45 train/test splits
for train_idx, test_idx in cv.split(X):
    model.fit(X[train_idx], y[train_idx])
    ...
```

### Validated CV (CPCV + DSR in one step)

Combines CPCV with Deflated Sharpe Ratio to assess whether your
strategy's performance survives multiple-testing correction:

```python
from ml4t.diagnostic import ValidatedCrossValidation
from ml4t.diagnostic.config import ValidatedCrossValidationConfig

config = ValidatedCrossValidationConfig(
    n_groups=10,
    n_test_groups=2,
    embargo_pct=0.01,
    annualization_factor=252,
)

vcv = ValidatedCrossValidation(config=config)
result = vcv.fit_evaluate(X, y, model, times=dates)

print(f"Mean Sharpe: {result.mean_sharpe:.2f}")
print(f"DSR: {result.dsr:.2f}")
print(f"Significant: {result.is_significant}")
for line in result.interpretation:
    print(f"  {line}")
```

### Visualize folds

```python
from ml4t.diagnostic.visualization import plot_cv_folds

fig = plot_cv_folds(cv, X, theme="default")
fig.show()
```

---

## Choosing a Workflow

| I have... | I want... | Start here |
|-----------|-----------|-----------|
| Factor + prices | Signal quality assessment | [Signal Analysis](#1-signal-analysis) |
| Strategy returns | Performance metrics & dashboard | [Portfolio Analysis](#2-portfolio-analysis) |
| Fitted model + features | Feature ranking | [Feature Importance](#3-feature-importance) |
| Losing trades + model | Understand why trades fail | [Trade Error Analysis](#4-trade-error-analysis-trade-shap) |
| Time-series ML data | Leakage-free evaluation | [Cross-Validation](#5-cross-validation) |
| Many candidate signals | Multiple-testing correction | [Multi-signal comparison](#multi-signal-comparison-with-fdr-correction) |
| Sharpe ratios from trials | Backtest overfitting check | [Validated CV](#validated-cv-cpcv-dsr-in-one-step) |
