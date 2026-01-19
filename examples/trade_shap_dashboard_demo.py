"""Demo script for Trade-SHAP Dashboard.

This script demonstrates how to launch the Trade-SHAP diagnostics dashboard
with mock data or real TradeShapResult objects.

Usage:
    # Run with streamlit (standalone mode - file upload)
    streamlit run examples/trade_shap_dashboard_demo.py

    # Run programmatically (with data)
    python examples/trade_shap_dashboard_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_mock_result() -> dict:
    """Create mock TradeShapResult for demonstration.

    Returns
    -------
    dict
        Mock result dictionary compatible with dashboard
    """
    import random
    from datetime import datetime, timedelta

    import numpy as np

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Mock error patterns with hypotheses and actions
    mock_patterns = [
        {
            "cluster_id": 0,
            "n_trades": 15,
            "description": "High momentum (↑0.45) + Low volatility (↓-0.32) → Losses",
            "top_features": [
                ("momentum_20d", 0.45, 0.001, 0.002, True),
                ("volatility_10d", -0.32, 0.003, 0.004, True),
                ("rsi_14", 0.28, 0.010, 0.012, True),
            ],
            "separation_score": 1.2,
            "distinctiveness": 1.8,
            "hypothesis": "Trades entering low-volatility momentum trends that reverse quickly",
            "actions": [
                "Add volatility filter: avoid trades when volatility < 10th percentile",
                "Consider mean-reversion features to detect trend exhaustion",
                "Add volume confirmation to validate momentum strength",
            ],
            "confidence": 0.85,
        },
        {
            "cluster_id": 1,
            "n_trades": 22,
            "description": "High RSI (↑0.38) + High volume (↑0.29) → Losses",
            "top_features": [
                ("rsi_14", 0.38, 0.001, 0.001, True),
                ("volume_ratio", 0.29, 0.005, 0.006, True),
                ("price_change_1d", 0.21, 0.020, 0.022, False),
            ],
            "separation_score": 0.9,
            "distinctiveness": 1.5,
            "hypothesis": "Trades entering overbought conditions with high volume (potential reversals)",
            "actions": [
                "Add overbought filter: skip trades when RSI > 70",
                "Consider volume profile: avoid high volume in overbought zones",
                "Add mean reversion features to capture reversal dynamics",
            ],
            "confidence": 0.78,
        },
        {
            "cluster_id": 2,
            "n_trades": 8,
            "description": "Low liquidity (↓-0.52) + Wide spread (↑0.41) → Losses",
            "top_features": [
                ("bid_ask_spread", 0.41, 0.002, 0.003, True),
                ("market_depth", -0.52, 0.001, 0.001, True),
                ("trading_volume", -0.35, 0.008, 0.009, True),
            ],
            "separation_score": 1.5,
            "distinctiveness": 2.1,
            "hypothesis": "Poor execution quality in illiquid markets",
            "actions": [
                "Add liquidity filter: minimum market depth threshold",
                "Consider spread cost in signal generation",
                "Use limit orders instead of market orders in low-liquidity conditions",
            ],
            "confidence": 0.92,
        },
    ]

    # Generate realistic trade data
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "MATIC-USD"]
    feature_sets = [
        [("momentum_20d", 0.45), ("volatility_10d", -0.32), ("rsi_14", 0.28)],
        [("rsi_14", 0.38), ("volume_ratio", 0.29), ("price_change_1d", 0.21)],
        [("market_depth", -0.52), ("bid_ask_spread", 0.41), ("trading_volume", -0.35)],
        [("bollinger_upper", 0.35), ("macd_signal", -0.28), ("adx_14", 0.22)],
    ]

    start_date = datetime(2024, 1, 1)
    mock_explanations = []

    # Generate 50 trades with realistic metrics
    for i in range(50):
        # Random symbol
        symbol = random.choice(symbols)

        # Random timestamp
        days_offset = random.randint(0, 180)
        hours_offset = random.randint(0, 23)
        timestamp = start_date + timedelta(days=days_offset, hours=hours_offset)

        # Random feature set
        top_features = random.choice(feature_sets).copy()

        # Generate realistic trade metrics (mostly losses for "worst trades")
        # 80% losses, 20% wins (to simulate worst trades analysis)
        is_loss = random.random() < 0.8

        if is_loss:
            pnl = random.uniform(-1500, -50)  # Losses
            return_pct = random.uniform(-5.0, -0.5)
        else:
            pnl = random.uniform(50, 800)  # Wins
            return_pct = random.uniform(0.5, 4.0)

        entry_price = random.uniform(1000, 50000)
        exit_price = entry_price * (1 + return_pct / 100)
        duration_days = random.uniform(0.5, 10.0)

        # Generate feature values at trade entry
        feature_values = {}
        for feat_name, shap_val in top_features:
            # Generate realistic feature values based on name
            if "momentum" in feat_name.lower():
                feature_values[feat_name] = random.uniform(-0.5, 2.0)
            elif "volatility" in feat_name.lower():
                feature_values[feat_name] = random.uniform(0.001, 0.05)
            elif "rsi" in feat_name.lower():
                feature_values[feat_name] = random.uniform(20, 80)
            elif "volume" in feat_name.lower():
                feature_values[feat_name] = random.uniform(0.5, 5.0)
            elif "spread" in feat_name.lower():
                feature_values[feat_name] = random.uniform(0.0001, 0.01)
            elif "depth" in feat_name.lower():
                feature_values[feat_name] = random.uniform(100, 10000)
            elif "bollinger" in feat_name.lower():
                feature_values[feat_name] = random.uniform(0.8, 1.2)
            elif "macd" in feat_name.lower():
                feature_values[feat_name] = random.uniform(-0.02, 0.02)
            elif "adx" in feat_name.lower():
                feature_values[feat_name] = random.uniform(10, 50)
            else:
                feature_values[feat_name] = random.uniform(-1, 1)

        mock_explanations.append(
            {
                "trade_id": f"{symbol}_{timestamp.strftime('%Y%m%d_%H%M')}",
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "top_features": top_features,
                "feature_values": feature_values,
                "trade_metrics": {
                    "symbol": symbol,
                    "pnl": pnl,
                    "return_pct": return_pct,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "duration_days": duration_days,
                },
            }
        )

    # Sort by PnL (worst first)
    mock_explanations.sort(key=lambda x: x["trade_metrics"]["pnl"])  # type: ignore[index]

    return {
        "n_trades_analyzed": 50,
        "n_trades_explained": 45,
        "n_trades_failed": 5,
        "explanations": mock_explanations,
        "failed_trades": [
            ("TRADE_046", "Missing SHAP values at timestamp"),
            ("TRADE_047", "Feature alignment failed"),
            ("TRADE_048", "Timestamp outside feature range"),
            ("TRADE_049", "NaN values in features"),
            ("TRADE_050", "SHAP computation error"),
        ],
        "error_patterns": mock_patterns,
    }


def main():
    """Main entry point for programmatic usage."""
    try:
        from ml4t.diagnostic.evaluation.trade_shap_dashboard import (
            run_diagnostics_dashboard,
        )
    except ImportError as e:
        print(f"❌ Failed to import dashboard: {e}")
        print("\nInstall streamlit with: pip install streamlit")
        print("Or install with: pip install ml4t-diagnostic[dashboard]")
        sys.exit(1)

    # Create mock result
    print("Creating mock Trade-SHAP result...")
    mock_result = create_mock_result()

    print("✅ Mock result created:")
    print(f"   - {mock_result['n_trades_analyzed']} trades analyzed")
    print(f"   - {mock_result['n_trades_explained']} successfully explained")
    print(f"   - {len(mock_result['error_patterns'])} error patterns identified")

    print("\n🚀 Launching dashboard...")
    print("   Press Ctrl+C to stop\n")

    # Launch dashboard with mock data
    run_diagnostics_dashboard(
        result=mock_result,
        title="Trade-SHAP Diagnostics - Demo Dashboard",
    )


if __name__ == "__main__":
    main()
