# Installation

## Requirements

- Python 3.11+
- Polars 0.20+

## Basic Installation

```bash
pip install ml4t-diagnostic
```

## Optional Dependencies

ML4T Diagnostic has optional dependency groups for different use cases:

### Machine Learning Features

For SHAP-based feature importance and trade error analysis:

```bash
pip install ml4t-diagnostic[ml]
```

Includes: `lightgbm`, `xgboost`, `shap`

### Visualization

For interactive Plotly charts and report generation:

```bash
pip install ml4t-diagnostic[viz]
```

Includes: `plotly`, `matplotlib`, `kaleido`

### Advanced Statistics

For ARCH/GARCH volatility models:

```bash
pip install ml4t-diagnostic[advanced]
```

Includes: `arch`

### Full Installation

Install all optional dependencies:

```bash
pip install ml4t-diagnostic[all]
```

## Development Installation

For contributing to ML4T Diagnostic:

```bash
git clone https://github.com/stefan-jansen/ml4t-diagnostic
cd ml4t-diagnostic
pip install -e ".[all,dev]"
```

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
