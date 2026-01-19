# Trade Diagnostics Example Notebook

## Overview

`trade_diagnostics_example.ipynb` is a comprehensive end-to-end tutorial demonstrating the complete ml4t-diagnostic workflow for analyzing and improving ML trading strategies.

## What's Included

This notebook covers the **complete diagnostic workflow**:

1. **Synthetic Backtest Generation** - Creates realistic trading data with intentional error patterns
2. **Trade Analysis** - Identifies worst/best trades and computes statistics
3. **Statistical Validation (DSR)** - Accounts for multiple testing bias
4. **SHAP Explanations** - Explains why specific trades failed
5. **Error Pattern Clustering** - Discovers recurring failure modes
6. **Hypothesis Generation** - Generates actionable improvement suggestions
7. **Dashboard Integration** - Shows how to use the interactive Streamlit dashboard
8. **Implementation Roadmap** - Provides concrete next steps

## Quick Start

### Option 1: Local Jupyter

```bash
cd /home/stefan/ml4t/software/evaluation

# Install dependencies
pip install ml4t-diagnostic[ml]

# Launch Jupyter
jupyter notebook examples/trade_diagnostics_example.ipynb
```

### Option 2: JupyterLab

```bash
# Install JupyterLab if needed
pip install jupyterlab

# Launch
jupyter lab examples/trade_diagnostics_example.ipynb
```

### Option 3: VS Code

Open the notebook in VS Code with the Jupyter extension installed.

## Prerequisites

**Required**:
- Python 3.9+
- ml4t-diagnostic with ML dependencies: `pip install ml4t-diagnostic[ml]`

**Optional (for dashboard)**:
- Streamlit: `pip install ml4t-diagnostic[dashboard]`

## Notebook Structure

**23 total cells** (10 markdown + 13 code):

### Section 1: Setup and Imports
- Import required libraries
- Configure visualization settings
- Verify installation

### Section 2: Generate Synthetic Backtest Data
- Creates 75 realistic trades over 6 months
- 3 intentional error patterns:
  - Pattern 1: High momentum + Low volatility → Reversals
  - Pattern 2: Low liquidity + Wide spreads → Poor execution
  - Pattern 3: Regime changes + Correlation breaks → Failures
- 10 features with realistic ranges
- SHAP values designed for clear clustering
- 40% win rate (realistic)

### Section 3: Basic Trade Analysis
- Compute win/loss statistics
- Calculate PnL metrics
- Visualize distribution and cumulative PnL

### Section 4: Statistical Validation (DSR)
- Calculate Deflated Sharpe Ratio
- Account for multiple testing bias (100 strategies tested)
- Interpret probability of true profitability
- Understand significance after selection bias correction

### Section 5: SHAP Analysis
- Initialize TradeShapAnalyzer
- Explain worst individual trade
- Visualize SHAP waterfall plot
- Identify top feature contributors

### Section 6: Error Pattern Clustering
- Cluster trades by SHAP similarity
- Discover 3 distinct error patterns
- Analyze pattern characteristics
- Visualize pattern distribution

### Section 7: Hypothesis Generation
- Generate actionable hypotheses for each pattern
- Get specific improvement recommendations
- Estimate potential savings from fixes
- Prioritize implementation

### Section 8: Dashboard Integration
- Check dashboard availability
- Instructions for launching Streamlit dashboard
- Overview of dashboard features

### Section 9: Summary & Next Steps
- Key insights from analysis
- 4-phase implementation roadmap
- Additional resources
- Academic references

## Expected Output

When you run the complete notebook, you'll see:

1. **Trade Statistics**:
   - ~40% win rate
   - Positive total PnL
   - Sharpe ratio ~0.8-1.2
   - Deflated SR accounting for multiple testing

2. **Visualizations**:
   - PnL distribution histogram
   - Cumulative PnL chart
   - SHAP waterfall for worst trade
   - Error pattern pie chart

3. **Diagnostic Insights**:
   - 3 distinct error patterns identified
   - Top contributing features for each pattern
   - Specific hypotheses about failure causes
   - Concrete action items for improvement

4. **Implementation Roadmap**:
   - Quick wins (1-2 weeks)
   - Feature engineering (2-4 weeks)
   - Validation (2-3 weeks)
   - Production deployment (ongoing)

## Testing

A test script is included to verify notebook components work correctly:

```bash
cd /home/stefan/ml4t/software/evaluation
.venv/bin/python examples/test_notebook_execution.py
```

This tests:
- All imports
- Synthetic data generation
- Trade analysis
- DSR calculation
- SHAP analyzer initialization
- Trade explanation

## Key Features Demonstrated

### 1. Realistic Synthetic Data
- Intentional error patterns that mimic real trading failures
- Consistent PnL calculation (passes TradeRecord validation)
- Realistic feature ranges for crypto futures
- SHAP values designed for interpretability

### 2. Complete Workflow
- End-to-end from raw trades to actionable insights
- All major ml4t-diagnostic capabilities
- Integration between components
- Real-world application patterns

### 3. Best Practices
- Proper random seed setting for reproducibility
- Clear cell-by-cell explanations
- Professional visualizations
- Academic rigor (correct DSR, SHAP usage)

### 4. Production-Ready Patterns
- Error handling
- Type consistency
- API usage examples
- Dashboard integration

## Common Issues & Solutions

### Issue: "No module named 'pandas'"
**Solution**: Install ml4t-diagnostic with ML dependencies:
```bash
pip install ml4t-diagnostic[ml]
```

### Issue: "TradeRecord validation error"
**Solution**: The notebook generates PnL consistently with prices. If you modify the data generation, ensure:
```python
if direction == "long":
    pnl = (exit_price - entry_price) * quantity
else:
    pnl = (entry_price - exit_price) * quantity
```

### Issue: "KeyError: 'deflated_sr'"
**Solution**: The DSR function returns keys `dsr`, `dsr_zscore`, not `deflated_sr`, `z_score`. Use:
```python
dsr_result['dsr']  # The deflated Sharpe ratio
dsr_result['dsr_zscore']  # The z-score
```

### Issue: "AttributeError: 'TradeStatistics' object has no attribute 'sharpe_ratio'"
**Solution**: Calculate Sharpe ratio manually:
```python
returns = np.array([t.pnl for t in trades])
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
```

## Next Steps After Running Notebook

1. **Adapt to Your Data**:
   - Replace synthetic data with your actual backtest results
   - Ensure data follows TradeRecord schema
   - Provide your feature matrix and SHAP values

2. **Customize Analysis**:
   - Adjust `n_clusters` for your dataset size
   - Modify `n_worst` trades to analyze
   - Change `n_trials` for DSR to match your testing

3. **Explore Dashboard**:
   - Run `streamlit run examples/trade_shap_dashboard_demo.py`
   - Interactive exploration of patterns
   - Export visualizations

4. **Implement Improvements**:
   - Follow the generated hypotheses
   - Add recommended filters
   - Engineer new features
   - Retrain and validate

## Additional Resources

- **Main Documentation**: `/docs/DASHBOARD.md`
- **Dashboard Demo**: `examples/trade_shap_dashboard_demo.py`
- **API Reference**: Module docstrings in `ml4t.diagnostic.evaluation`
- **Integration Examples**: Other files in `examples/`

## Academic References

1. Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal of Portfolio Management*, 40(5), 94-107.

2. Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems*, 30.

3. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 6: Backtesting.

## License

This example is part of ml4t-diagnostic and follows the same license as the main library.

## Support

For issues or questions:
1. Check this README
2. Review the notebook comments
3. See main ml4t-diagnostic documentation
4. File an issue in the repository
