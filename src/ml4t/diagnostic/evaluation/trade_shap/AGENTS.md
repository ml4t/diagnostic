# evaluation/trade_shap/ - Trade-Level SHAP Pipeline

Building blocks for explaining losing trades and clustering recurring error patterns.

## Main Components

- `explain.py` - per-trade SHAP explanations
- `alignment.py` - timestamp alignment between trades and feature rows
- `fold_shap.py` - fold-aware SHAP extraction
- `normalize.py` - SHAP vector normalization helpers
- `cluster.py` - clustering and centroid utilities
- `characterize.py` - feature statistics and pattern characterization
- `hypotheses/` - template-driven hypothesis generation
- `pipeline.py` - `TradeShapPipeline`
- `models.py` - `TradeShapResult`, `TradeShapExplanation`, `ErrorPattern`

## Notes

- The user-facing wrapper class is `TradeShapAnalyzer` in `trade_shap_diagnostics.py`.
- Backtest tearsheets can render Trade-SHAP sections via `visualization/backtest/`.
