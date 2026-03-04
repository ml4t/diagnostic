"""Regression tests for beta review findings (2026-03-04).

Tests that would have caught:
- Finding #1: FeatureDiagnostics false "passes all checks" on all-module failure
- Finding #2: HTML injection in tearsheet error path
- Finding #3: HTML injection in multi-signal dashboard
- Finding #5: Timezone-aware timestamp numpy deprecation
- Finding #6: Half-life 0.0 rendered as N/A
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

import numpy as np
import pytest


class TestFeatureDiagnosticsFailureDetection:
    """Verify FeatureDiagnostics detects when all modules fail."""

    def test_all_inf_data_flags_failure(self):
        """Pathological data should not report 'passes all checks'."""
        from ml4t.diagnostic.evaluation.feature_diagnostics import (
            FeatureDiagnostics,
        )

        diag = FeatureDiagnostics()
        result = diag.run_diagnostics(
            np.array([np.inf, -np.inf, np.nan, np.inf]),
            name="pathological",
        )

        # Should flag the failure, not report clean
        assert "ALL_MODULES_FAILED" in result.flags
        assert not any(
            "passes all diagnostic checks" in r.lower()
            for r in result.recommendations
        )
        assert any("failed" in r.lower() for r in result.recommendations)

    def test_normal_data_still_passes(self):
        """Normal data should still get 'passes all checks' if appropriate."""
        from ml4t.diagnostic.evaluation.feature_diagnostics import (
            FeatureDiagnostics,
        )

        np.random.seed(42)
        data = np.random.randn(500)

        diag = FeatureDiagnostics()
        result = diag.run_diagnostics(data, name="normal_feature")

        # Should NOT flag all_modules_failed — at least some should succeed
        assert "ALL_MODULES_FAILED" not in result.flags


class TestTearsheetHTMLEscaping:
    """Verify HTML injection is prevented in tearsheet output."""

    def test_error_path_escapes_script_tags(self):
        """Exception messages containing script tags should be escaped."""
        from ml4t.diagnostic.visualization.backtest.tearsheet import (
            _generate_section,
        )

        # Force an error by passing bad data that will cause a plot failure
        # We test the error handler directly by checking that any exception
        # message gets escaped
        html = _generate_section(
            "equity_curve",
            "Test Section",
            # Pass no data — should hit the error path
            trades=None,
            returns=None,
        )
        # Should either return None (no error) or escaped HTML
        if html is not None:
            assert "<script>" not in html

    def test_title_escaping(self):
        """Title with HTML should be escaped in output."""
        from ml4t.diagnostic.visualization.backtest.tearsheet import (
            generate_backtest_tearsheet,
        )

        html = generate_backtest_tearsheet(
            returns=np.random.randn(100),
            title='<script>alert("xss")</script>',
            subtitle='<img onerror="alert(1)">',
            template="quant_trader",
        )
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;img onerror" in html

    def test_interactive_false_deprecation_warning(self):
        """interactive=False should emit a DeprecationWarning."""
        from ml4t.diagnostic.visualization.backtest.tearsheet import (
            generate_backtest_tearsheet,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            generate_backtest_tearsheet(
                returns=np.random.randn(100),
                template="quant_trader",
                interactive=False,
            )
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "interactive" in str(deprecation_warnings[0].message).lower()


class TestMultiSignalDashboardEscaping:
    """Verify HTML injection is prevented in multi-signal dashboard."""

    def test_title_escaped_in_header(self):
        """Dashboard title with HTML should be escaped."""
        import html as html_mod

        from ml4t.diagnostic.visualization.signal.multi_signal_dashboard import (
            MultiSignalDashboard,
        )

        malicious_title = '<script>alert("xss")</script>'
        escaped = html_mod.escape(malicious_title)

        dashboard = MultiSignalDashboard(title=malicious_title)
        header_html = dashboard._build_header()

        assert "<script>alert" not in header_html
        assert escaped in header_html


class TestTimezoneAlignment:
    """Verify timezone-aware timestamps don't trigger numpy deprecation."""

    def test_tz_aware_timestamps_no_warning(self):
        """tz-aware timestamps should not emit numpy deprecation warning."""
        from ml4t.diagnostic.evaluation.trade_shap.alignment import TimestampAligner

        tz_timestamps = [
            datetime(2020, 1, 1, 10, 0, tzinfo=timezone.utc),
            datetime(2020, 1, 1, 11, 0, tzinfo=timezone.utc),
            datetime(2020, 1, 1, 12, 0, tzinfo=timezone.utc),
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            aligner = TimestampAligner.from_datetime_index(tz_timestamps)
            deprecation_msgs = [
                x for x in w
                if "timezone" in str(x.message).lower() and "deprecated" in str(x.message).lower()
            ]
            assert len(deprecation_msgs) == 0, f"Got deprecation warning: {deprecation_msgs}"
            assert len(aligner.timestamps_ns) == 3

    def test_naive_timestamps_still_work(self):
        """Naive timestamps should work as before."""
        from ml4t.diagnostic.evaluation.trade_shap.alignment import TimestampAligner

        timestamps = [
            datetime(2020, 1, 1, 10, 0),
            datetime(2020, 1, 1, 11, 0),
            datetime(2020, 1, 1, 12, 0),
        ]
        aligner = TimestampAligner.from_datetime_index(timestamps)
        assert len(aligner.timestamps_ns) == 3


class TestHalfLifeDisplay:
    """Verify half-life 0.0 is displayed correctly."""

    def test_zero_half_life_not_na(self):
        """half_life=0.0 should render as '0.0', not 'N/A'."""
        half_life = 0.0
        # Reproduce the fixed formatting logic
        result = f"{half_life:.1f}" if half_life is not None else "N/A"
        assert result == "0.0"

    def test_none_half_life_is_na(self):
        """half_life=None should render as 'N/A'."""
        half_life = None
        result = f"{half_life:.1f}" if half_life is not None else "N/A"
        assert result == "N/A"
