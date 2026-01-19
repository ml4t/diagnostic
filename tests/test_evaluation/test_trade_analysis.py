"""Tests for trade analysis module.

Tests cover:
    - TradeMetrics: Creation, computed fields, DataFrame conversion
    - TradeAnalysis: Worst/best trade extraction, filtering
    - TradeStatistics: Aggregate metrics computation
    - TradeAnalysisResult: Serialization, DataFrame export
    - Edge cases: Empty trades, single trade, all winners/losers
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from pydantic import ValidationError

from ml4t.diagnostic.evaluation import (
    TradeAnalysis,
    TradeAnalysisResult,
    TradeMetrics,
    TradeStatistics,
)
from ml4t.diagnostic.integration.backtest_contract import TradeRecord


class TestTradeMetrics:
    """Tests for TradeMetrics Pydantic model."""

    def test_from_trade_record_long(self):
        """Test creating TradeMetrics from long trade."""
        # Arrange
        trade_record = TradeRecord(
            timestamp=datetime(2024, 1, 15, 10, 30),
            symbol="AAPL",
            entry_price=150.0,
            exit_price=155.0,
            pnl=500.0,
            duration=timedelta(days=5),
            direction="long",
            quantity=100.0,
        )

        # Act
        metrics = TradeMetrics.from_trade_record(trade_record)

        # Assert
        assert metrics.symbol == "AAPL"
        assert metrics.pnl == 500.0
        assert metrics.return_pct == pytest.approx(0.0333, abs=0.0001)  # (155-150)/150
        assert metrics.duration_hours == pytest.approx(120.0)  # 5 days * 24
        assert metrics.duration_days == pytest.approx(5.0)
        assert metrics.pnl_per_day == pytest.approx(100.0)  # 500/5

    def test_from_trade_record_short(self):
        """Test creating TradeMetrics from short trade."""
        # Arrange
        trade_record = TradeRecord(
            timestamp=datetime(2024, 2, 1, 14, 0),
            symbol="BTC-USD",
            entry_price=45000.0,
            exit_price=44000.0,
            pnl=1000.0,
            duration=timedelta(hours=6),
            direction="short",
            quantity=1.0,
        )

        # Act
        metrics = TradeMetrics.from_trade_record(trade_record)

        # Assert
        assert metrics.symbol == "BTC-USD"
        assert metrics.pnl == 1000.0
        # Short: (entry - exit) / entry = (45000 - 44000) / 45000
        assert metrics.return_pct == pytest.approx(0.0222, abs=0.0001)
        assert metrics.duration_hours == pytest.approx(6.0)
        assert metrics.duration_days == pytest.approx(0.25)
        assert metrics.pnl_per_day == pytest.approx(4000.0)  # 1000/0.25

    def test_computed_fields_losingstrade(self):
        """Test computed fields for losing trade."""
        # Arrange
        trade_record = TradeRecord(
            timestamp=datetime(2024, 3, 1),
            symbol="TSLA",
            entry_price=200.0,
            exit_price=190.0,
            pnl=-1000.0,
            duration=timedelta(days=2),
            direction="long",
            quantity=100.0,
        )

        # Act
        metrics = TradeMetrics.from_trade_record(trade_record)

        # Assert
        assert metrics.pnl == -1000.0
        assert metrics.return_pct == pytest.approx(-0.05)  # (190-200)/200
        assert metrics.pnl_per_day == pytest.approx(-500.0)  # -1000/2

    def test_to_dict(self):
        """Test dictionary export."""
        # Arrange
        trade_record = TradeRecord(
            timestamp=datetime(2024, 1, 15),
            symbol="AAPL",
            entry_price=150.0,
            exit_price=155.0,
            pnl=500.0,
            duration=timedelta(days=5),
            direction="long",
        )
        metrics = TradeMetrics.from_trade_record(trade_record)

        # Act
        data = metrics.to_dict()

        # Assert
        assert data["symbol"] == "AAPL"
        assert data["pnl"] == 500.0
        assert "duration_seconds" in data
        assert data["duration_seconds"] == 5 * 86400.0  # 5 days in seconds
        assert "return_pct" in data
        assert "duration_hours" in data
        assert "duration_days" in data
        assert "pnl_per_day" in data

    def test_to_dataframe_single(self):
        """Test DataFrame conversion with single trade."""
        # Arrange
        trade_record = TradeRecord(
            timestamp=datetime(2024, 1, 15),
            symbol="AAPL",
            entry_price=150.0,
            exit_price=155.0,
            pnl=500.0,
            duration=timedelta(days=5),
            direction="long",
        )
        metrics = TradeMetrics.from_trade_record(trade_record)

        # Act
        df = TradeMetrics.to_dataframe([metrics])

        # Assert
        assert isinstance(df, pl.DataFrame)
        assert df.height == 1
        assert "symbol" in df.columns
        assert "pnl" in df.columns
        assert "return_pct" in df.columns
        assert "duration_hours" in df.columns
        assert "pnl_per_day" in df.columns
        assert df["symbol"][0] == "AAPL"
        assert df["pnl"][0] == 500.0

    def test_to_dataframe_multiple(self):
        """Test DataFrame conversion with multiple trades."""
        # Arrange
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, i),
                symbol=f"SYM{i}",
                entry_price=100.0,
                exit_price=100.0 + i,
                pnl=float(i * 100),
                duration=timedelta(days=i),
                direction="long",
            )
            for i in range(1, 6)
        ]
        metrics_list = [TradeMetrics.from_trade_record(t) for t in trades]

        # Act
        df = TradeMetrics.to_dataframe(metrics_list)

        # Assert
        assert isinstance(df, pl.DataFrame)
        assert df.height == 5
        assert df["symbol"].to_list() == ["SYM1", "SYM2", "SYM3", "SYM4", "SYM5"]
        assert df["pnl"].to_list() == [100.0, 200.0, 300.0, 400.0, 500.0]

    def test_to_dataframe_empty(self):
        """Test DataFrame conversion with empty list."""
        # Act
        df = TradeMetrics.to_dataframe([])

        # Assert
        assert isinstance(df, pl.DataFrame)
        assert df.height == 0
        # Should have expected schema
        assert "symbol" in df.columns
        assert "pnl" in df.columns
        assert "return_pct" in df.columns

    def test_validation_negative_duration(self):
        """Test validation rejects negative duration."""
        with pytest.raises(ValidationError, match="Duration must be positive"):
            TradeMetrics(
                timestamp=datetime(2024, 1, 15),
                symbol="AAPL",
                entry_price=150.0,
                exit_price=155.0,
                pnl=500.0,
                duration=timedelta(days=-1),  # Invalid
            )


class TestTradeStatistics:
    """Tests for TradeStatistics aggregate metrics."""

    def test_compute_basic_stats(self):
        """Test basic statistics computation."""
        # Arrange - 3 winners, 2 losers
        trades = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol="SYM1",
                entry_price=100.0,
                exit_price=110.0,
                pnl=1000.0,
                duration=timedelta(days=1),
            )
            for i in range(1, 4)  # 3 winners
        ] + [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol="SYM2",
                entry_price=100.0,
                exit_price=95.0,
                pnl=-500.0,
                duration=timedelta(days=2),
            )
            for i in range(4, 6)  # 2 losers
        ]

        # Act
        stats = TradeStatistics.compute(trades)

        # Assert
        assert stats.n_trades == 5
        assert stats.n_winners == 3
        assert stats.n_losers == 2
        assert stats.win_rate == pytest.approx(0.6)  # 3/5
        assert stats.total_pnl == pytest.approx(2000.0)  # 3*1000 - 2*500
        assert stats.avg_pnl == pytest.approx(400.0)  # 2000/5
        assert stats.avg_winner == pytest.approx(1000.0)
        assert stats.avg_loser == pytest.approx(-500.0)

    def test_compute_profit_factor(self):
        """Test profit factor calculation."""
        # Arrange
        trades = [
            TradeMetrics(
                timestamp=datetime(2024, 1, 1),
                symbol="WIN",
                entry_price=100.0,
                exit_price=120.0,
                pnl=2000.0,  # Gross profit
                duration=timedelta(days=1),
            ),
            TradeMetrics(
                timestamp=datetime(2024, 1, 2),
                symbol="LOSS",
                entry_price=100.0,
                exit_price=90.0,
                pnl=-1000.0,  # Gross loss
                duration=timedelta(days=1),
            ),
        ]

        # Act
        stats = TradeStatistics.compute(trades)

        # Assert
        # Profit factor = gross_profit / gross_loss = 2000 / 1000 = 2.0
        assert stats.profit_factor == pytest.approx(2.0)

    def test_compute_all_winners(self):
        """Test statistics with all winning trades."""
        # Arrange
        trades = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol="WIN",
                entry_price=100.0,
                exit_price=110.0,
                pnl=1000.0,
                duration=timedelta(days=1),
            )
            for i in range(1, 11)  # 10 winners
        ]

        # Act
        stats = TradeStatistics.compute(trades)

        # Assert
        assert stats.n_trades == 10
        assert stats.n_winners == 10
        assert stats.n_losers == 0
        assert stats.win_rate == 1.0
        assert stats.avg_winner == pytest.approx(1000.0)
        assert stats.avg_loser is None  # No losers
        assert stats.profit_factor is None  # No losers, can't divide by zero

    def test_compute_all_losers(self):
        """Test statistics with all losing trades."""
        # Arrange
        trades = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol="LOSS",
                entry_price=100.0,
                exit_price=90.0,
                pnl=-500.0,
                duration=timedelta(days=1),
            )
            for i in range(1, 11)  # 10 losers
        ]

        # Act
        stats = TradeStatistics.compute(trades)

        # Assert
        assert stats.n_trades == 10
        assert stats.n_winners == 0
        assert stats.n_losers == 10
        assert stats.win_rate == 0.0
        assert stats.avg_winner is None  # No winners
        assert stats.avg_loser == pytest.approx(-500.0)
        assert stats.profit_factor is None  # No winners

    def test_compute_single_trade(self):
        """Test statistics with single trade."""
        # Arrange
        trades = [
            TradeMetrics(
                timestamp=datetime(2024, 1, 1),
                symbol="SINGLE",
                entry_price=100.0,
                exit_price=110.0,
                pnl=1000.0,
                duration=timedelta(days=5),
            )
        ]

        # Act
        stats = TradeStatistics.compute(trades)

        # Assert
        assert stats.n_trades == 1
        assert stats.n_winners == 1
        assert stats.n_losers == 0
        assert stats.win_rate == 1.0
        assert stats.total_pnl == 1000.0
        assert stats.avg_pnl == 1000.0
        assert stats.avg_duration_days == pytest.approx(5.0)
        assert stats.median_duration_days == pytest.approx(5.0)

    def test_compute_empty_raises(self):
        """Test that empty trade list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute statistics for empty trade list"):
            TradeStatistics.compute([])

    def test_quartiles(self):
        """Test PnL and duration quartile calculations."""
        # Arrange - Create trades with known PnL distribution
        trades = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol="SYM",
                entry_price=100.0,
                exit_price=100.0 + i * 10,
                pnl=float(i * 100),  # 100, 200, 300, 400, 500
                duration=timedelta(days=i),  # 1, 2, 3, 4, 5 days
            )
            for i in range(1, 6)
        ]

        # Act
        stats = TradeStatistics.compute(trades)

        # Assert
        # PnL: 100, 200, 300, 400, 500
        assert stats.pnl_quartiles["q50"] == pytest.approx(300.0)  # Median
        # Duration: 1, 2, 3, 4, 5 days
        assert stats.duration_quartiles["q50"] == pytest.approx(3.0)  # Median

    def test_summary_output(self):
        """Test summary string generation."""
        # Arrange
        trades = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol="SYM",
                entry_price=100.0,
                exit_price=105.0,
                pnl=500.0,
                duration=timedelta(days=2),
            )
            for i in range(1, 6)
        ]

        # Act
        stats = TradeStatistics.compute(trades)
        summary = stats.summary()

        # Assert
        assert "Trade Statistics" in summary
        assert "Total trades: 5" in summary
        assert "Win rate:" in summary
        assert "Total PnL:" in summary
        assert "Average PnL:" in summary

    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        # Arrange
        trades = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol="SYM",
                entry_price=100.0,
                exit_price=105.0,
                pnl=500.0,
                duration=timedelta(days=2),
            )
            for i in range(1, 6)
        ]
        stats = TradeStatistics.compute(trades)

        # Act
        df = stats.to_dataframe()

        # Assert
        assert isinstance(df, pl.DataFrame)
        assert df.height == 1  # Single row with all stats
        assert "n_trades" in df.columns
        assert "win_rate" in df.columns
        assert "total_pnl" in df.columns
        assert df["n_trades"][0] == 5


class TestTradeAnalysis:
    """Tests for TradeAnalysis class."""

    def test_init_basic(self):
        """Test basic initialization."""
        # Arrange
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, i),
                symbol="SYM",
                entry_price=100.0,
                exit_price=105.0,
                pnl=500.0,
                duration=timedelta(days=1),
            )
            for i in range(1, 11)
        ]

        # Act
        analyzer = TradeAnalysis(trades)

        # Assert
        assert len(analyzer.trades) == 10
        assert all(isinstance(t, TradeMetrics) for t in analyzer.trades)

    def test_init_empty_raises(self):
        """Test that empty trade list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot analyze empty trade list"):
            TradeAnalysis([])

    def test_worst_trades(self):
        """Test extraction of worst trades."""
        # Arrange - Create trades with known PnL
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, i),
                symbol=f"SYM{i}",
                entry_price=100.0,
                exit_price=100.0 + (i - 5) * 10,  # Centered around 0
                pnl=float((i - 5) * 100),  # -400, -300, ..., 0, ..., 400
                duration=timedelta(days=1),
            )
            for i in range(1, 10)
        ]

        # Act
        analyzer = TradeAnalysis(trades)
        worst = analyzer.worst_trades(n=3)

        # Assert
        assert len(worst) == 3
        # Should be sorted by PnL ascending
        assert worst[0].pnl == -400.0  # (1-5)*100
        assert worst[1].pnl == -300.0  # (2-5)*100
        assert worst[2].pnl == -200.0  # (3-5)*100

    def test_best_trades(self):
        """Test extraction of best trades."""
        # Arrange
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, i),
                symbol=f"SYM{i}",
                entry_price=100.0,
                exit_price=100.0 + i * 10,
                pnl=float(i * 100),  # 100, 200, ..., 900
                duration=timedelta(days=1),
            )
            for i in range(1, 10)
        ]

        # Act
        analyzer = TradeAnalysis(trades)
        best = analyzer.best_trades(n=3)

        # Assert
        assert len(best) == 3
        # Should be sorted by PnL descending
        assert best[0].pnl == 900.0
        assert best[1].pnl == 800.0
        assert best[2].pnl == 700.0

    def test_worst_trades_n_exceeds_total(self):
        """Test worst_trades when N > total trades."""
        # Arrange
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, i),
                symbol="SYM",
                entry_price=100.0,
                exit_price=105.0,
                pnl=500.0,
                duration=timedelta(days=1),
            )
            for i in range(1, 6)  # Only 5 trades
        ]

        # Act
        analyzer = TradeAnalysis(trades)
        worst = analyzer.worst_trades(n=10)  # Request 10

        # Assert
        assert len(worst) == 5  # Should return all available

    def test_worst_trades_invalid_n(self):
        """Test that invalid n raises ValueError."""
        # Arrange
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, 1),
                symbol="SYM",
                entry_price=100.0,
                exit_price=105.0,
                pnl=500.0,
                duration=timedelta(days=1),
            )
        ]
        analyzer = TradeAnalysis(trades)

        # Act & Assert
        with pytest.raises(ValueError, match="n must be positive"):
            analyzer.worst_trades(n=0)

        with pytest.raises(ValueError, match="n must be positive"):
            analyzer.worst_trades(n=-1)

    def test_compute_statistics(self):
        """Test statistics computation."""
        # Arrange
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, i),
                symbol="SYM",
                entry_price=100.0,
                exit_price=105.0 if i % 2 == 0 else 95.0,  # Alternating wins/losses
                pnl=500.0 if i % 2 == 0 else -500.0,
                duration=timedelta(days=1),
            )
            for i in range(1, 11)  # 10 trades
        ]

        # Act
        analyzer = TradeAnalysis(trades)
        stats = analyzer.compute_statistics()

        # Assert
        assert isinstance(stats, TradeStatistics)
        assert stats.n_trades == 10
        assert stats.n_winners == 5
        assert stats.n_losers == 5
        assert stats.win_rate == pytest.approx(0.5)

    def test_analyze_full_workflow(self):
        """Test complete analyze() workflow."""
        # Arrange
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, i),
                symbol=f"SYM{i}",
                entry_price=100.0,
                exit_price=100.0 + (i - 5) * 10,
                pnl=float((i - 5) * 100),
                duration=timedelta(days=1),
            )
            for i in range(1, 11)  # 10 trades: -400 to +500
        ]

        # Act
        analyzer = TradeAnalysis(trades)
        result = analyzer.analyze(n_worst=3, n_best=2)

        # Assert
        assert isinstance(result, TradeAnalysisResult)
        assert len(result.worst_trades) == 3
        assert len(result.best_trades) == 2
        assert result.n_total_trades == 10
        assert result.worst_trades[0].pnl == -400.0
        assert result.best_trades[0].pnl == 500.0

    def test_filtering_by_symbol(self):
        """Test symbol-based filtering."""
        # Arrange
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, i),
                symbol="AAPL" if i <= 5 else "MSFT",
                entry_price=100.0,
                exit_price=105.0,
                pnl=500.0,
                duration=timedelta(days=1),
            )
            for i in range(1, 11)  # 5 AAPL, 5 MSFT
        ]
        filter_config = {"symbols": ["AAPL"]}

        # Act
        analyzer = TradeAnalysis(trades, filter_config=filter_config)

        # Assert
        assert len(analyzer.trades) == 5
        assert all(t.symbol == "AAPL" for t in analyzer.trades)

    def test_filtering_by_duration(self):
        """Test duration-based filtering."""
        # Arrange
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, i),
                symbol="SYM",
                entry_price=100.0,
                exit_price=105.0,
                pnl=500.0,
                duration=timedelta(hours=i),  # 1h, 2h, ..., 10h
            )
            for i in range(1, 11)
        ]
        filter_config = {"min_duration_seconds": 3600 * 5}  # 5 hours minimum

        # Act
        analyzer = TradeAnalysis(trades, filter_config=filter_config)

        # Assert
        assert len(analyzer.trades) == 6  # Hours 5-10
        assert all(t.duration.total_seconds() >= 3600 * 5 for t in analyzer.trades)

    def test_filtering_empty_result_raises(self):
        """Test that filtering to zero trades raises ValueError."""
        # Arrange
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, i),
                symbol="AAPL",
                entry_price=100.0,
                exit_price=105.0,
                pnl=500.0,
                duration=timedelta(days=1),
            )
            for i in range(1, 6)
        ]
        filter_config = {"symbols": ["MSFT"]}  # No matches

        # Act & Assert
        with pytest.raises(ValueError, match="No trades remaining after applying filters"):
            TradeAnalysis(trades, filter_config=filter_config)


class TestTradeAnalysisResult:
    """Tests for TradeAnalysisResult schema."""

    def test_init_basic(self):
        """Test basic initialization."""
        # Arrange
        worst = [
            TradeMetrics(
                timestamp=datetime(2024, 1, 1),
                symbol="LOSS",
                entry_price=100.0,
                exit_price=90.0,
                pnl=-1000.0,
                duration=timedelta(days=1),
            )
        ]
        best = [
            TradeMetrics(
                timestamp=datetime(2024, 1, 2),
                symbol="WIN",
                entry_price=100.0,
                exit_price=120.0,
                pnl=2000.0,
                duration=timedelta(days=1),
            )
        ]
        stats = TradeStatistics.compute(worst + best)

        # Act
        result = TradeAnalysisResult(
            worst_trades=worst,
            best_trades=best,
            statistics=stats,
            n_total_trades=10,
        )

        # Assert
        assert len(result.worst_trades) == 1
        assert len(result.best_trades) == 1
        assert result.n_total_trades == 10
        assert result.analysis_type == "trade_analysis"

    def test_to_json_string(self):
        """Test JSON serialization."""
        # Arrange
        worst = [
            TradeMetrics(
                timestamp=datetime(2024, 1, 1),
                symbol="SYM",
                entry_price=100.0,
                exit_price=95.0,
                pnl=-500.0,
                duration=timedelta(days=1),
            )
        ]
        best = [
            TradeMetrics(
                timestamp=datetime(2024, 1, 2),
                symbol="SYM",
                entry_price=100.0,
                exit_price=110.0,
                pnl=1000.0,
                duration=timedelta(days=1),
            )
        ]
        stats = TradeStatistics.compute(worst + best)
        result = TradeAnalysisResult(
            worst_trades=worst,
            best_trades=best,
            statistics=stats,
            n_total_trades=5,
        )

        # Act
        json_str = result.to_json_string()

        # Assert
        assert isinstance(json_str, str)
        assert "worst_trades" in json_str
        assert "best_trades" in json_str
        assert "statistics" in json_str

    def test_get_dataframe_worst_trades(self):
        """Test worst_trades DataFrame export."""
        # Arrange
        worst = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol=f"SYM{i}",
                entry_price=100.0,
                exit_price=90.0,
                pnl=float(-100 * i),
                duration=timedelta(days=1),
            )
            for i in range(1, 4)
        ]
        best = [
            TradeMetrics(
                timestamp=datetime(2024, 1, 10),
                symbol="WIN",
                entry_price=100.0,
                exit_price=110.0,
                pnl=1000.0,
                duration=timedelta(days=1),
            )
        ]
        stats = TradeStatistics.compute(worst + best)
        result = TradeAnalysisResult(
            worst_trades=worst,
            best_trades=best,
            statistics=stats,
            n_total_trades=10,
        )

        # Act
        df = result.get_dataframe("worst_trades")

        # Assert
        assert isinstance(df, pl.DataFrame)
        assert df.height == 3
        assert "symbol" in df.columns
        assert "pnl" in df.columns

    def test_get_dataframe_statistics(self):
        """Test statistics DataFrame export."""
        # Arrange
        trades = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol="SYM",
                entry_price=100.0,
                exit_price=105.0,
                pnl=500.0,
                duration=timedelta(days=1),
            )
            for i in range(1, 6)
        ]
        stats = TradeStatistics.compute(trades)
        result = TradeAnalysisResult(
            worst_trades=trades[:2],
            best_trades=trades[-2:],
            statistics=stats,
            n_total_trades=10,
        )

        # Act
        df = result.get_dataframe("statistics")

        # Assert
        assert isinstance(df, pl.DataFrame)
        assert df.height == 1  # Single row
        assert "n_trades" in df.columns
        assert "win_rate" in df.columns

    def test_get_dataframe_all_trades(self):
        """Test all_trades DataFrame export (combined worst + best)."""
        # Arrange
        worst = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol=f"WORST{i}",
                entry_price=100.0,
                exit_price=90.0,
                pnl=-500.0,
                duration=timedelta(days=1),
            )
            for i in range(1, 3)
        ]
        best = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol=f"BEST{i}",
                entry_price=100.0,
                exit_price=120.0,
                pnl=2000.0,
                duration=timedelta(days=1),
            )
            for i in range(1, 3)
        ]
        stats = TradeStatistics.compute(worst + best)
        result = TradeAnalysisResult(
            worst_trades=worst,
            best_trades=best,
            statistics=stats,
            n_total_trades=10,
        )

        # Act
        df = result.get_dataframe("all_trades")

        # Assert
        assert isinstance(df, pl.DataFrame)
        assert df.height == 4  # 2 worst + 2 best

    def test_get_dataframe_invalid_name(self):
        """Test that invalid DataFrame name raises ValueError."""
        # Arrange
        trades = [
            TradeMetrics(
                timestamp=datetime(2024, 1, 1),
                symbol="SYM",
                entry_price=100.0,
                exit_price=105.0,
                pnl=500.0,
                duration=timedelta(days=1),
            )
        ]
        stats = TradeStatistics.compute(trades)
        result = TradeAnalysisResult(
            worst_trades=trades,
            best_trades=trades,
            statistics=stats,
            n_total_trades=10,
        )

        # Act & Assert
        with pytest.raises(ValueError, match="DataFrame 'invalid' not available"):
            result.get_dataframe("invalid")

    def test_list_available_dataframes(self):
        """Test listing available DataFrames."""
        # Arrange
        trades = [
            TradeMetrics(
                timestamp=datetime(2024, 1, 1),
                symbol="SYM",
                entry_price=100.0,
                exit_price=105.0,
                pnl=500.0,
                duration=timedelta(days=1),
            )
        ]
        stats = TradeStatistics.compute(trades)
        result = TradeAnalysisResult(
            worst_trades=trades,
            best_trades=trades,
            statistics=stats,
            n_total_trades=10,
        )

        # Act
        available = result.list_available_dataframes()

        # Assert
        assert available == ["worst_trades", "best_trades", "statistics", "all_trades"]

    def test_summary_output(self):
        """Test summary string generation."""
        # Arrange
        worst = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol=f"LOSS{i}",
                entry_price=100.0,
                exit_price=90.0,
                pnl=float(-100 * i),
                duration=timedelta(days=1),
            )
            for i in range(1, 6)
        ]
        best = [
            TradeMetrics(
                timestamp=datetime(2024, 1, i),
                symbol=f"WIN{i}",
                entry_price=100.0,
                exit_price=120.0,
                pnl=float(200 * i),
                duration=timedelta(days=1),
            )
            for i in range(1, 6)
        ]
        stats = TradeStatistics.compute(worst + best)
        result = TradeAnalysisResult(
            worst_trades=worst,
            best_trades=best,
            statistics=stats,
            n_total_trades=20,
        )

        # Act
        summary = result.summary()

        # Assert
        assert "Trade Analysis Summary" in summary
        assert "Total trades analyzed: 20" in summary
        assert "Worst trades extracted: 5" in summary
        assert "Best trades extracted: 5" in summary
        assert "Win rate:" in summary
        assert "Worst Trades (Top 5)" in summary
        assert "Best Trades (Top 5)" in summary


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_realistic(self):
        """Test complete workflow with realistic synthetic data."""
        # Arrange - Create realistic trade data
        import random

        random.seed(42)

        trades = []
        for i in range(1, 101):  # 100 trades
            is_winner = random.random() > 0.4  # 60% win rate
            entry = 100.0
            exit_ = entry * (1 + random.gauss(0.05 if is_winner else -0.03, 0.02))
            quantity = random.uniform(50, 200)
            pnl = (exit_ - entry) * quantity

            trades.append(
                TradeRecord(
                    timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
                    symbol=random.choice(["AAPL", "MSFT", "GOOGL", "TSLA"]),
                    entry_price=entry,
                    exit_price=exit_,
                    pnl=pnl,
                    duration=timedelta(hours=random.randint(1, 24)),
                    direction="long",
                    quantity=quantity,
                )
            )

        # Act - Complete workflow
        analyzer = TradeAnalysis(trades)
        result = analyzer.analyze(n_worst=20, n_best=10)

        # Assert
        assert result.n_total_trades == 100
        assert len(result.worst_trades) == 20
        assert len(result.best_trades) == 10
        assert result.statistics.n_trades == 100
        # Should be roughly 60% win rate (with randomness)
        assert 0.5 <= result.statistics.win_rate <= 0.7

        # Test serialization
        json_str = result.to_json_string()
        assert isinstance(json_str, str)

        # Test DataFrame export
        df_worst = result.get_dataframe("worst_trades")
        assert df_worst.height == 20

        # Test summary
        summary = result.summary()
        assert "100" in summary  # Total trades

    def test_edge_case_single_trade(self):
        """Test handling of single trade."""
        # Arrange
        trades = [
            TradeRecord(
                timestamp=datetime(2024, 1, 1),
                symbol="ONLY",
                entry_price=100.0,
                exit_price=110.0,
                pnl=1000.0,
                duration=timedelta(days=5),
            )
        ]

        # Act
        analyzer = TradeAnalysis(trades)
        result = analyzer.analyze(n_worst=1, n_best=1)

        # Assert
        assert result.n_total_trades == 1
        assert len(result.worst_trades) == 1
        assert len(result.best_trades) == 1
        # Same trade appears in both
        assert result.worst_trades[0].pnl == 1000.0
        assert result.best_trades[0].pnl == 1000.0
