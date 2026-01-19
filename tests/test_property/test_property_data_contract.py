"""Property-based tests for data contract models using Hypothesis."""

from datetime import datetime

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ml4t.diagnostic.integration.data_contract import (
    AnomalyType,
    DataAnomaly,
    DataQualityMetrics,
    DataQualityReport,
    DataValidationRequest,
    Severity,
)


# Custom strategies
@st.composite
def valid_metrics(draw):
    """Generate valid DataQualityMetrics."""
    completeness = draw(st.floats(min_value=0.0, max_value=1.0))
    timeliness = draw(st.floats(min_value=0.0, max_value=1000.0))
    accuracy = draw(st.floats(min_value=0.0, max_value=1.0))
    consistency = draw(st.floats(min_value=0.0, max_value=1.0))
    n_records = draw(st.integers(min_value=0, max_value=10_000_000))
    n_anomalies = draw(st.integers(min_value=0, max_value=1000))
    n_critical = draw(st.integers(min_value=0, max_value=min(n_anomalies, 100)))
    n_error = draw(st.integers(min_value=0, max_value=min(n_anomalies - n_critical, 500)))
    n_warning = draw(st.integers(min_value=0, max_value=n_anomalies - n_critical - n_error))

    return DataQualityMetrics(
        completeness=completeness,
        timeliness=timeliness,
        accuracy_score=accuracy,
        consistency_score=consistency,
        n_records=n_records,
        n_anomalies=n_anomalies,
        n_critical=n_critical,
        n_error=n_error,
        n_warning=n_warning,
    )


@st.composite
def valid_anomaly(draw):
    """Generate valid DataAnomaly."""
    anomaly_type = draw(st.sampled_from(list(AnomalyType)))
    severity = draw(st.sampled_from(list(Severity)))
    timestamp = draw(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2030, 12, 31)))
    symbol = draw(st.one_of(st.none(), st.text(min_size=1, max_size=10)))
    description = draw(st.text(min_size=1, max_size=200))
    value = draw(st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)))

    return DataAnomaly(
        anomaly_type=anomaly_type,
        severity=severity,
        timestamp=timestamp,
        symbol=symbol,
        description=description,
        value=value,
    )


class TestDataQualityMetricsProperties:
    """Property-based tests for DataQualityMetrics."""

    @given(valid_metrics())
    @settings(max_examples=100)
    def test_metrics_roundtrip(self, metrics):
        """Test that metrics can be serialized and deserialized."""
        data = metrics.model_dump()
        restored = DataQualityMetrics(**data)

        assert restored.completeness == metrics.completeness
        assert restored.n_records == metrics.n_records
        assert restored.n_anomalies == metrics.n_anomalies

    @given(
        completeness=st.floats(min_value=0.0, max_value=1.0),
        n_records=st.integers(min_value=0, max_value=1_000_000),
    )
    @settings(max_examples=50)
    def test_completeness_bounds(self, completeness, n_records):
        """Test completeness is always in [0, 1]."""
        metrics = DataQualityMetrics(
            completeness=completeness,
            timeliness=0.0,
            accuracy_score=1.0,
            consistency_score=1.0,
            n_records=n_records,
            n_anomalies=0,
        )
        assert 0.0 <= metrics.completeness <= 1.0


class TestDataAnomalyProperties:
    """Property-based tests for DataAnomaly."""

    @given(valid_anomaly())
    @settings(max_examples=100)
    def test_anomaly_roundtrip(self, anomaly):
        """Test that anomalies can be serialized and deserialized."""
        data = anomaly.model_dump()
        restored = DataAnomaly(**data)

        assert restored.anomaly_type == anomaly.anomaly_type
        assert restored.severity == anomaly.severity
        assert restored.description == anomaly.description

    @given(
        anomaly_type=st.sampled_from(list(AnomalyType)),
        severity=st.sampled_from(list(Severity)),
    )
    @settings(max_examples=50)
    def test_enum_values(self, anomaly_type, severity):
        """Test that enum values are correctly stored."""
        anomaly = DataAnomaly(
            anomaly_type=anomaly_type,
            severity=severity,
            timestamp=datetime.now(),
            description="Test anomaly",
        )
        assert anomaly.anomaly_type == anomaly_type
        assert anomaly.severity == severity
        # String comparison also works
        assert anomaly.anomaly_type.value == anomaly_type.value


class TestDataQualityReportProperties:
    """Property-based tests for DataQualityReport."""

    @given(valid_metrics())
    @settings(max_examples=50)
    def test_is_acceptable_monotonic(self, metrics):
        """Test that relaxing thresholds makes acceptance more likely."""
        report = DataQualityReport(
            symbol="TEST",
            source="test",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=metrics,
            is_production_ready=True,
        )

        # Strict thresholds
        strict = report.is_acceptable(min_completeness=0.99, max_critical=0, max_errors=0)
        # Relaxed thresholds
        relaxed = report.is_acceptable(min_completeness=0.5, max_critical=100, max_errors=1000)

        # If strict passes, relaxed must pass (monotonicity)
        if strict:
            assert relaxed

    @given(valid_metrics())
    @settings(max_examples=50)
    def test_summary_contains_key_info(self, metrics):
        """Test that summary contains essential information."""
        report = DataQualityReport(
            symbol="AAPL",
            source="test_source",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=metrics,
            is_production_ready=False,
        )
        summary = report.summary()

        # Must contain symbol
        assert "AAPL" in summary
        # Must contain source
        assert "test_source" in summary
        # Must contain production ready status
        assert "NO" in summary or "YES" in summary

    @given(valid_metrics())
    @settings(max_examples=50)
    def test_to_dict_json_serializable(self, metrics):
        """Test that to_dict output is JSON serializable."""
        import json

        report = DataQualityReport(
            symbol="TEST",
            source="test",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1d",
            metrics=metrics,
            is_production_ready=True,
        )
        data = report.to_dict()

        # Must not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

        # Round trip
        parsed = json.loads(json_str)
        assert parsed["symbol"] == "TEST"


class TestDataValidationRequestProperties:
    """Property-based tests for DataValidationRequest."""

    @given(
        symbol=st.text(min_size=1, max_size=10),
        checks=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10),
    )
    @settings(max_examples=50)
    def test_request_roundtrip(self, symbol, checks):
        """Test request serialization roundtrip."""
        assume(symbol.strip())  # Non-empty after stripping

        # Use checks if provided, otherwise let default apply
        if checks:
            request = DataValidationRequest(symbol=symbol, checks=checks)
        else:
            request = DataValidationRequest(symbol=symbol)
        data = request.model_dump()
        restored = DataValidationRequest(**data)

        assert restored.symbol == request.symbol


class TestCrossModuleProperties:
    """Property-based tests across multiple models."""

    @given(
        n_anomalies=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=30)
    def test_report_anomaly_count_matches(self, n_anomalies):
        """Test that anomaly list length matches metrics."""
        anomalies = [
            DataAnomaly(
                anomaly_type=AnomalyType.OUTLIER,
                severity=Severity.WARNING,
                timestamp=datetime.now(),
                description=f"Anomaly {i}",
            )
            for i in range(n_anomalies)
        ]

        metrics = DataQualityMetrics(
            completeness=0.95,
            timeliness=1.0,
            accuracy_score=0.99,
            consistency_score=1.0,
            n_records=1000,
            n_anomalies=n_anomalies,
            n_warning=n_anomalies,
        )

        report = DataQualityReport(
            symbol="TEST",
            source="test",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1d",
            metrics=metrics,
            anomalies=anomalies,
            is_production_ready=n_anomalies == 0,
        )

        assert len(report.anomalies) == n_anomalies
        assert report.metrics.n_anomalies == n_anomalies
