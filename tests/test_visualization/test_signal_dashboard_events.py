"""Tests for SignalDashboard Events tab integration.

Tests that the SignalDashboard properly renders EventStudyResult
in the Events tab with all visualizations.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ml4t.diagnostic.config.event_config import EventConfig, WindowSettings
from ml4t.diagnostic.evaluation.event_analysis import EventStudyAnalysis
from ml4t.diagnostic.visualization.signal import SignalDashboard

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trading_dates() -> list[datetime]:
    """Generate 500 trading days."""
    start = datetime(2020, 1, 1)
    dates = []
    current = start
    while len(dates) < 500:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


@pytest.fixture
def sample_event_study_result(trading_dates: list[datetime]):
    """Create sample EventStudyResult for dashboard testing."""
    np.random.seed(42)
    n_assets = 10
    assets = [f"ASSET_{i:02d}" for i in range(n_assets)]

    # Create returns data
    records = []
    for asset in assets:
        for date in trading_dates:
            ret = np.random.normal(0.0005, 0.02)
            records.append({"date": date, "asset": asset, "return": ret})

    returns_df = pl.DataFrame(records)

    # Create benchmark data
    benchmark_df = pl.DataFrame(
        {
            "date": trading_dates,
            "return": np.random.normal(0.0003, 0.015, len(trading_dates)),
        }
    )

    # Create events (5 events across different assets and dates)
    event_dates = [
        trading_dates[300],
        trading_dates[320],
        trading_dates[340],
        trading_dates[360],
        trading_dates[380],
    ]
    events_df = pl.DataFrame(
        {
            "date": event_dates,
            "asset": ["ASSET_00", "ASSET_01", "ASSET_02", "ASSET_03", "ASSET_04"],
        }
    )

    config = EventConfig(
        window=WindowSettings(
            estimation_start=-252,
            estimation_end=-20,
            event_start=-5,
            event_end=5,
        ),
        model="market_model",
        test="t_test",
        min_estimation_obs=100,
    )

    analysis = EventStudyAnalysis(
        returns=returns_df,
        events=events_df,
        benchmark=benchmark_df,
        config=config,
    )

    return analysis.run()


# =============================================================================
# Tests
# =============================================================================


class TestSignalDashboardEventsTab:
    """Tests for Events tab in SignalDashboard."""

    def test_dashboard_generates_with_events_tab(
        self,
        sample_event_study_result,
    ) -> None:
        """Test that dashboard generates HTML with Events tab."""
        dashboard = SignalDashboard(title="Test Dashboard")

        # Create minimal tear sheet mock
        from unittest.mock import MagicMock

        mock_tear_sheet = MagicMock()
        mock_tear_sheet.signal_name = "test_signal"
        mock_tear_sheet.n_assets = 10
        mock_tear_sheet.n_dates = 100
        mock_tear_sheet.date_range = ("2020-01-01", "2020-12-31")
        mock_tear_sheet.ic_analysis = None
        mock_tear_sheet.quantile_analysis = None
        mock_tear_sheet.turnover_analysis = None
        mock_tear_sheet.ir_tc_analysis = None

        html = dashboard.generate(
            mock_tear_sheet,
            include_events=True,
            event_analysis=sample_event_study_result,
        )

        # Verify HTML contains Events tab
        assert isinstance(html, str)
        assert len(html) > 1000
        assert "Events" in html
        assert "Event Study Analysis" in html

    def test_events_tab_contains_summary_metrics(
        self,
        sample_event_study_result,
    ) -> None:
        """Test that Events tab contains summary metrics."""
        dashboard = SignalDashboard()

        # Generate events tab HTML directly
        events_html = dashboard._create_events_tab(sample_event_study_result)

        # Check for key metrics
        assert "Events Analyzed" in events_html
        assert str(sample_event_study_result.n_events) in events_html
        assert "Event Window" in events_html
        assert "Final CAAR" in events_html
        assert "P-value" in events_html
        assert sample_event_study_result.test_name.replace("_", " ").title() in events_html

    def test_events_tab_contains_caar_plot(
        self,
        sample_event_study_result,
    ) -> None:
        """Test that Events tab contains CAAR visualization."""
        dashboard = SignalDashboard()

        events_html = dashboard._create_events_tab(sample_event_study_result)

        # Should contain plotly div (CAAR plot embedded)
        assert "plotly" in events_html.lower() or "js-plotly" in events_html.lower()
        # Should have the CAAR title
        assert "CAAR" in events_html

    def test_events_tab_contains_event_heatmap(
        self,
        sample_event_study_result,
    ) -> None:
        """Test that Events tab contains event heatmap."""
        dashboard = SignalDashboard()

        events_html = dashboard._create_events_tab(sample_event_study_result)

        # Should have heatmap-related content
        assert "Abnormal Return" in events_html or "AR" in events_html

    def test_events_tab_contains_events_table(
        self,
        sample_event_study_result,
    ) -> None:
        """Test that Events tab contains individual events table."""
        dashboard = SignalDashboard()

        events_html = dashboard._create_events_tab(sample_event_study_result)

        # Should have table with event details
        assert "Individual Event Results" in events_html
        assert "<table" in events_html
        assert "Event ID" in events_html
        assert "Asset" in events_html
        assert "CAR" in events_html

    def test_events_tab_with_dark_theme(
        self,
        sample_event_study_result,
    ) -> None:
        """Test that Events tab works with dark theme."""
        dashboard = SignalDashboard(theme="dark")

        events_html = dashboard._create_events_tab(sample_event_study_result)

        # Should generate without error
        assert isinstance(events_html, str)
        assert len(events_html) > 500

    def test_events_tab_significance_badge(
        self,
        sample_event_study_result,
    ) -> None:
        """Test that significance badge is displayed correctly."""
        dashboard = SignalDashboard()

        events_html = dashboard._create_events_tab(sample_event_study_result)

        # Should show either "Significant" or "Not Significant"
        assert "Significant" in events_html or "Not Significant" in events_html

    def test_dashboard_save_with_events(
        self,
        sample_event_study_result,
        tmp_path,
    ) -> None:
        """Test that dashboard can be saved with Events tab."""
        from unittest.mock import MagicMock

        dashboard = SignalDashboard(title="Test Dashboard with Events")

        # Create minimal tear sheet mock
        mock_tear_sheet = MagicMock()
        mock_tear_sheet.signal_name = "test_signal"
        mock_tear_sheet.n_assets = 10
        mock_tear_sheet.n_dates = 100
        mock_tear_sheet.date_range = ("2020-01-01", "2020-12-31")
        mock_tear_sheet.ic_analysis = None
        mock_tear_sheet.quantile_analysis = None
        mock_tear_sheet.turnover_analysis = None
        mock_tear_sheet.ir_tc_analysis = None

        output_path = tmp_path / "dashboard_with_events.html"

        dashboard.save(
            str(output_path),
            mock_tear_sheet,
            include_events=True,
            event_analysis=sample_event_study_result,
        )

        # Verify file was created
        assert output_path.exists()
        content = output_path.read_text()
        assert "Event Study Analysis" in content
        assert len(content) > 10000  # Should be substantial HTML

    def test_events_table_car_color_coding(
        self,
        sample_event_study_result,
    ) -> None:
        """Test that CAR values are color-coded (green for positive, red for negative)."""
        dashboard = SignalDashboard()

        table_html = dashboard._create_events_table(sample_event_study_result)

        # Should have color styling for positive/negative CARs
        assert "#28a745" in table_html or "#dc3545" in table_html  # green or red

    def test_events_tab_without_individual_results(self) -> None:
        """Test that Events tab handles missing individual results gracefully."""
        from ml4t.diagnostic.results.event_results import EventStudyResult

        # Create minimal result without individual_results
        result = EventStudyResult(
            aar_by_day={-5: 0.001, 0: 0.025, 5: -0.002},
            caar=[0.001, 0.015, 0.05, 0.045, 0.048, 0.073, 0.068, 0.065, 0.060, 0.055, 0.050],
            caar_dates=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
            caar_std=[0.01] * 11,
            caar_ci_lower=[-0.01] * 11,
            caar_ci_upper=[0.1] * 11,
            test_statistic=2.5,
            p_value=0.02,
            test_name="t_test",
            n_events=10,
            model_name="market_model",
            event_window=(-5, 5),
            confidence_level=0.95,
            individual_results=None,  # No individual results
        )

        dashboard = SignalDashboard()
        events_html = dashboard._create_events_tab(result)

        # Should generate without error, but skip heatmap/table
        assert isinstance(events_html, str)
        assert "Event Study Analysis" in events_html
        assert "Events Analyzed" in events_html
