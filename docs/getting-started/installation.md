# Installation

## Requirements

- Python 3.12+
- Polars 0.20+

## Basic Installation

```bash
pip install ml4t-diagnostic
```

## Optional Dependencies

ML4T Diagnostic has optional dependency groups for different use cases:

### Visualization

For Plotly charts, tearsheets, and PDF export:

```bash
pip install ml4t-diagnostic[viz]
```

Includes: `plotly`, `matplotlib`, `seaborn`, `kaleido`, `pypdf`

### Machine Learning Backends

For LightGBM and XGBoost model analysis:

```bash
pip install ml4t-diagnostic[ml]
```

Includes: `lightgbm`, `xgboost`

### Backtest Bridge

For `ml4t-backtest` integration and result-to-tearsheet bridges:

```bash
pip install ml4t-diagnostic[backtest]
```

Includes: `ml4t-backtest`

### Dashboard

For the optional Streamlit trade diagnostics dashboard:

```bash
pip install ml4t-diagnostic[dashboard]
```

Includes: `streamlit`

### Full Installation

Install all optional dependencies:

```bash
pip install ml4t-diagnostic[all]
```

## Development Installation

For contributing to ML4T Diagnostic:

```bash
git clone https://github.com/ml4t/diagnostic.git
cd ml4t-diagnostic
pip install -e ".[all,dev]"
```

## Using The Book Code Locally

If you are running the third-edition notebooks or case studies against a local checkout,
install the library in editable mode so the book code sees your current branch:

```bash
uv pip install -e /path/to/ml4t-diagnostic
```

See the [Book Guide](../book-guide/index.md) for the chapter and case-study map.
For the new reporting bridge, see the [Backtest Tearsheets](../user-guide/backtest-tearsheets.md) guide.

## Verify Installation

```python
import ml4t.diagnostic as diag
print(diag.__version__)
```

## Dependencies

### Core

| Package | Version | Purpose |
|---------|---------|---------|
| polars | ≥0.20.0 | Primary data processing |
| pandas | ≥2.0.0 | Compatibility layer |
| numpy | ≥1.24.0 | Numerical computing |
| scipy | ≥1.10.0 | Scientific computing |
| scikit-learn | ≥1.3.0 | ML utilities |
| statsmodels | ≥0.14.0 | Statistical tests |
| numba | ≥0.57.0 | JIT compilation |

### Optional

| Package | Group | Purpose |
|---------|-------|---------|
| lightgbm | ml | Gradient boosting |
| xgboost | ml | Gradient boosting |
| shap | ml | SHAP explanations |
| plotly | viz | Interactive charts |
| matplotlib | viz | Static charts |
| arch | advanced | GARCH models |
