#!/usr/bin/env python3
"""Test script to verify the trade diagnostics notebook can execute.

This validates that:
1. All imports work
2. Synthetic data generation succeeds
3. Core analysis functions execute
4. No runtime errors
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("TESTING TRADE DIAGNOSTICS NOTEBOOK COMPONENTS")
print("=" * 70)

# Test 1: Imports
print("\n[1/6] Testing imports...")
try:
    from datetime import datetime, timedelta

    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import polars as pl

    from ml4t.diagnostic.evaluation import TradeAnalysis, TradeShapAnalyzer
    from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio
    from ml4t.diagnostic.integration.backtest_contract import TradeRecord

    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Synthetic data generation
print("\n[2/6] Testing synthetic data generation...")
try:
    np.random.seed(42)

    # Simplified version of notebook's data generation
    n_trades = 30  # Smaller for testing
    feature_names = [
        "momentum_5d",
        "volatility_20d",
        "rsi_14",
        "volume_ratio",
        "trend_strength",
        "liquidity",
        "correlation",
        "skewness",
        "kurtosis",
        "regime_prob",
    ]

    trades = []
    features_list = []
    shap_list = []

    start_date = datetime(2024, 1, 1)

    for i in range(n_trades):
        # Generate trade
        timestamp = start_date + timedelta(days=i * 2)
        entry_price = np.random.uniform(10000, 50000)
        return_pct = np.random.uniform(-3, 3)
        exit_price = entry_price * (1 + return_pct / 100)
        direction = "long"
        quantity = 1.0

        # Calculate PnL based on prices (must be consistent)
        if direction == "long":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        trade = TradeRecord(
            timestamp=timestamp,
            symbol="BTC-PERP",
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            duration=timedelta(days=np.random.uniform(1, 5)),
            direction=direction,
            quantity=quantity,
        )
        trades.append(trade)

        # Generate features
        features = [np.random.uniform(-1, 1) for _ in feature_names]
        features_list.append(features)

        # Generate SHAP values
        shap_vals = [np.random.uniform(-0.5, 0.5) for _ in feature_names]
        shap_list.append(shap_vals)

    features_array = np.array(features_list)
    shap_array = np.array(shap_list)

    features_df = pl.DataFrame(
        {
            **{"timestamp": [t.timestamp for t in trades]},
            **{name: features_array[:, i] for i, name in enumerate(feature_names)},
        }
    )

    print(f"✅ Generated {len(trades)} trades")
    print(f"   Features shape: {features_df.shape}")
    print(f"   SHAP shape: {shap_array.shape}")

except Exception as e:
    print(f"❌ Data generation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 3: Trade analysis
print("\n[3/6] Testing TradeAnalysis...")
try:
    analyzer = TradeAnalysis(trades)
    worst_trades = analyzer.worst_trades(n=10)
    best_trades = analyzer.best_trades(n=5)
    stats = analyzer.compute_statistics()

    print("✅ Trade analysis successful")
    print(f"   Total trades: {stats.n_trades}")
    print(f"   Win rate: {stats.win_rate:.1%}")
    # Note: TradeStatistics doesn't have sharpe_ratio - would need TradeAnalysisResult

except Exception as e:
    print(f"❌ Trade analysis failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: DSR calculation
print("\n[4/6] Testing Deflated Sharpe Ratio...")
try:
    returns = np.array([t.pnl for t in trades]) / 100000
    skewness = float(pd.Series(returns).skew())
    kurtosis = float(pd.Series(returns).kurtosis() + 3)

    # Calculate Sharpe ratio manually
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0

    dsr_result = deflated_sharpe_ratio(
        observed_sharpe=sharpe_ratio,
        n_trials=100,
        variance_trials=0.15,
        n_samples=len(returns),
        skewness=skewness,
        kurtosis=kurtosis,
        return_components=True,
        return_format="probability",
    )

    print("✅ DSR calculation successful")
    print(f"   Observed SR: {sharpe_ratio:.3f}")
    print(f"   Deflated SR: {dsr_result['dsr']:.3f}")

except Exception as e:
    print(f"❌ DSR calculation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: SHAP analyzer
print("\n[5/6] Testing TradeShapAnalyzer...")
try:
    mock_model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)

    shap_analyzer = TradeShapAnalyzer(
        model=mock_model, features_df=features_df, shap_values=shap_array
    )

    print("✅ TradeShapAnalyzer initialized")

except Exception as e:
    print(f"❌ SHAP analyzer failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 6: Single trade explanation
print("\n[6/6] Testing trade explanation...")
try:
    worst_trade = worst_trades[0]
    explanation = shap_analyzer.explain_trade(worst_trade)

    print("✅ Trade explanation successful")
    print(f"   Worst trade PnL: ${worst_trade.pnl:,.2f}")
    print("   Top 3 features:")
    for feat, shap_val in explanation.top_features[:3]:
        print(f"     {feat:20s} {shap_val:+.3f}")

except Exception as e:
    print(f"❌ Trade explanation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Success
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED")
print("=" * 70)
print("\nThe notebook components work correctly!")
print("You can now run the full notebook: trade_diagnostics_example.ipynb")
