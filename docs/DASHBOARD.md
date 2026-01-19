# Trade-SHAP Dashboard

Interactive Streamlit dashboard for Trade-SHAP analysis visualization and systematic trade debugging.

## Overview

The Trade-SHAP Dashboard provides real-time visualization and analysis of ML trading strategy performance, using SHAP (SHapley Additive exPlanations) to explain why individual trades succeeded or failed.

## Features

### 1. Statistical Validation Tab
- DSR (Deflated Sharpe Ratio) analysis
- PSR (Probabilistic Sharpe Ratio) metrics
- Multiple testing correction results
- Backtest overfitting detection

### 2. Worst Trades Analysis Tab
- Interactive table of worst-performing trades
- Trade-by-trade SHAP explanations
- Feature contribution breakdown
- Export to CSV functionality

### 3. SHAP Analysis Tab
- Feature importance rankings
- SHAP value distributions
- Feature interaction effects
- Waterfall plots for individual trades

### 4. Error Patterns Tab
- Clustered error patterns
- Pattern descriptions and hypotheses
- Confidence scores
- Actionable improvement suggestions

## Installation

```bash
pip install ml4t-diagnostic[dashboard]
```

Or install Streamlit separately:

```bash
pip install streamlit
```

## Usage

### Command Line

```bash
# Run with default settings
streamlit run -m ml4t.diagnostic.evaluation.trade_shap_dashboard

# Or use the module directly
python -m streamlit run path/to/trade_shap_dashboard.py
```

### Programmatic

```python
from ml4t.diagnostic.evaluation import TradeShapAnalyzer
from ml4t.diagnostic.evaluation.trade_shap_dashboard import run_diagnostics_dashboard

# Analyze trades
analyzer = TradeShapAnalyzer(model, features_df, shap_values)
result = analyzer.explain_worst_trades(worst_trades)

# Launch dashboard
run_diagnostics_dashboard(result)
```

### From Saved Results

```python
from ml4t.diagnostic.evaluation.trade_shap_dashboard import (
    run_diagnostics_dashboard,
    load_data_from_file
)

# Load previously saved results
result = load_data_from_file("analysis_results.json")

# Launch dashboard
run_diagnostics_dashboard(result)
```

## Data Export

### Export Trades to CSV

```python
from ml4t.diagnostic.evaluation.trade_shap_dashboard import export_trades_to_csv

csv_content = export_trades_to_csv(trades_data)
with open("trades.csv", "w") as f:
    f.write(csv_content)
```

### Export Patterns to CSV

```python
from ml4t.diagnostic.evaluation.trade_shap_dashboard import export_patterns_to_csv

csv_content = export_patterns_to_csv(patterns)
with open("patterns.csv", "w") as f:
    f.write(csv_content)
```

### Export Full HTML Report

```python
from ml4t.diagnostic.evaluation.trade_shap_dashboard import export_full_report_html

html_content = export_full_report_html(result)
with open("report.html", "w") as f:
    f.write(html_content)
```

## Helper Functions

### Extract Trade Returns

```python
from ml4t.diagnostic.evaluation.trade_shap_dashboard import extract_trade_returns

returns = extract_trade_returns(result)
# Returns: list of PnL values
```

### Extract Trade Data

```python
from ml4t.diagnostic.evaluation.trade_shap_dashboard import extract_trade_data

trades = extract_trade_data(result)
# Returns: list of trade dictionaries with keys:
# - trade_id, timestamp, symbol, pnl, return_pct
# - duration_days, entry_price, exit_price
# - top_feature, top_feature_impact
```

## Architecture

The dashboard is implemented as a modular package:

```
ml4t.diagnostic.evaluation.trade_dashboard/
├── app.py          # Main Streamlit application
├── tabs/           # Tab modules
│   ├── stat_validation.py
│   ├── worst_trades.py
│   ├── shap_analysis.py
│   └── patterns.py
├── stats.py        # Statistical computations
├── export/         # Export functionality
│   ├── csv.py
│   └── html.py
├── io.py           # Data loading utilities
└── types.py        # Type definitions
```

## Requirements

- Python 3.9+
- Streamlit >= 1.0.0
- ml4t-diagnostic (core library)
- SHAP (for explanations)

## See Also

- [Trade-SHAP Analysis Guide](trade_shap.md)
- [Statistical Validation](validation.md)
- [SHAP Feature Importance](shap_importance.md)
