"""Tests for ml4t.diagnostic.integration.data_contract module."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from ml4t.diagnostic.integration.data_contract import (
    AnomalyType,
    DataAnomaly,
    DataQualityMetrics,
    DataQualityReport,
    DataValidationRequest,
    Severity,
)


class TestAnomalyType:
    """Tests for AnomalyType enum."""

    def test_anomaly_types_exist(self):
        """Test all expected anomaly types exist."""
        expected = [
            "MISSING_DATA",
            "STALE_DATA",
            "PRICE_SPIKE",
            "NEGATIVE_PRICE",
            "ZERO_VOLUME",
            "OHLC_VIOLATION",
            "TIMESTAMP_GAP",
            "DUPLICATE_TIMESTAMP",
            "OUTLIER",
        ]
        for name in expected:
            assert hasattr(AnomalyType, name)

    def test_anomaly_type_values(self):
        """Test anomaly type string values."""
        assert AnomalyType.MISSING_DATA.value == "missing_data"
        assert AnomalyType.STALE_DATA.value == "stale_data"
        assert AnomalyType.PRICE_SPIKE.value == "price_spike"
        assert AnomalyType.NEGATIVE_PRICE.value == "negative_price"
        assert AnomalyType.ZERO_VOLUME.value == "zero_volume"
        assert AnomalyType.OHLC_VIOLATION.value == "ohlc_violation"
        assert AnomalyType.TIMESTAMP_GAP.value == "timestamp_gap"
        assert AnomalyType.DUPLICATE_TIMESTAMP.value == "duplicate_timestamp"
        assert AnomalyType.OUTLIER.value == "outlier"

    def test_anomaly_type_is_str_enum(self):
        """Test AnomalyType inherits from str."""
        assert isinstance(AnomalyType.MISSING_DATA, str)
        assert AnomalyType.MISSING_DATA == "missing_data"


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_levels_exist(self):
        """Test all severity levels exist."""
        assert hasattr(Severity, "INFO")
        assert hasattr(Severity, "WARNING")
        assert hasattr(Severity, "ERROR")
        assert hasattr(Severity, "CRITICAL")

    def test_severity_values(self):
        """Test severity string values."""
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"
        assert Severity.CRITICAL.value == "critical"

    def test_severity_is_str_enum(self):
        """Test Severity inherits from str."""
        assert isinstance(Severity.WARNING, str)
        assert Severity.ERROR == "error"


class TestDataAnomaly:
    """Tests for DataAnomaly model."""

    def test_create_minimal_anomaly(self):
        """Test creating anomaly with required fields only."""
        anomaly = DataAnomaly(
            anomaly_type=AnomalyType.MISSING_DATA,
            severity=Severity.WARNING,
            timestamp=datetime(2024, 1, 15, 10, 30),
            description="Missing 5 trading days",
        )
        assert anomaly.anomaly_type == AnomalyType.MISSING_DATA
        assert anomaly.severity == Severity.WARNING
        assert anomaly.timestamp == datetime(2024, 1, 15, 10, 30)
        assert anomaly.description == "Missing 5 trading days"
        assert anomaly.symbol is None
        assert anomaly.value is None
        assert anomaly.expected_range is None
        assert anomaly.suggested_fix is None

    def test_create_full_anomaly(self):
        """Test creating anomaly with all fields."""
        anomaly = DataAnomaly(
            anomaly_type=AnomalyType.PRICE_SPIKE,
            severity=Severity.ERROR,
            timestamp=datetime(2024, 1, 15, 10, 30),
            symbol="AAPL",
            description="Price moved 15 std devs in 1 minute",
            value=999.99,
            expected_range=(150.0, 200.0),
            suggested_fix="Replace with interpolated value",
        )
        assert anomaly.symbol == "AAPL"
        assert anomaly.value == 999.99
        assert anomaly.expected_range == (150.0, 200.0)
        assert anomaly.suggested_fix == "Replace with interpolated value"

    def test_anomaly_type_validation(self):
        """Test that invalid anomaly_type is rejected."""
        with pytest.raises(ValidationError):
            DataAnomaly(
                anomaly_type="invalid_type",
                severity=Severity.WARNING,
                timestamp=datetime.now(),
                description="Test",
            )

    def test_severity_validation(self):
        """Test that invalid severity is rejected."""
        with pytest.raises(ValidationError):
            DataAnomaly(
                anomaly_type=AnomalyType.MISSING_DATA,
                severity="invalid_severity",
                timestamp=datetime.now(),
                description="Test",
            )

    def test_anomaly_serialization(self):
        """Test anomaly can be serialized to dict."""
        anomaly = DataAnomaly(
            anomaly_type=AnomalyType.STALE_DATA,
            severity=Severity.INFO,
            timestamp=datetime(2024, 1, 15),
            description="Test anomaly",
        )
        data = anomaly.model_dump()
        assert data["anomaly_type"] == "stale_data"
        assert data["severity"] == "info"
        assert "timestamp" in data
        assert data["description"] == "Test anomaly"


class TestDataQualityMetrics:
    """Tests for DataQualityMetrics model."""

    def test_create_metrics(self):
        """Test creating quality metrics."""
        metrics = DataQualityMetrics(
            completeness=0.98,
            timeliness=5.0,
            accuracy_score=0.95,
            consistency_score=1.0,
            n_records=10000,
            n_anomalies=12,
            n_critical=0,
            n_error=2,
            n_warning=10,
        )
        assert metrics.completeness == 0.98
        assert metrics.timeliness == 5.0
        assert metrics.accuracy_score == 0.95
        assert metrics.consistency_score == 1.0
        assert metrics.n_records == 10000
        assert metrics.n_anomalies == 12
        assert metrics.n_critical == 0
        assert metrics.n_error == 2
        assert metrics.n_warning == 10

    def test_metrics_defaults(self):
        """Test default values for optional fields."""
        metrics = DataQualityMetrics(
            completeness=1.0,
            timeliness=0.0,
            accuracy_score=1.0,
            consistency_score=1.0,
            n_records=100,
            n_anomalies=0,
        )
        assert metrics.n_critical == 0
        assert metrics.n_error == 0
        assert metrics.n_warning == 0

    def test_completeness_validation_min(self):
        """Test completeness must be >= 0."""
        with pytest.raises(ValidationError):
            DataQualityMetrics(
                completeness=-0.1,
                timeliness=0.0,
                accuracy_score=1.0,
                consistency_score=1.0,
                n_records=100,
                n_anomalies=0,
            )

    def test_completeness_validation_max(self):
        """Test completeness must be <= 1."""
        with pytest.raises(ValidationError):
            DataQualityMetrics(
                completeness=1.1,
                timeliness=0.0,
                accuracy_score=1.0,
                consistency_score=1.0,
                n_records=100,
                n_anomalies=0,
            )

    def test_timeliness_validation(self):
        """Test timeliness must be >= 0."""
        with pytest.raises(ValidationError):
            DataQualityMetrics(
                completeness=1.0,
                timeliness=-1.0,
                accuracy_score=1.0,
                consistency_score=1.0,
                n_records=100,
                n_anomalies=0,
            )

    def test_accuracy_validation(self):
        """Test accuracy_score must be in [0, 1]."""
        with pytest.raises(ValidationError):
            DataQualityMetrics(
                completeness=1.0,
                timeliness=0.0,
                accuracy_score=1.5,
                consistency_score=1.0,
                n_records=100,
                n_anomalies=0,
            )

    def test_n_records_validation(self):
        """Test n_records must be >= 0."""
        with pytest.raises(ValidationError):
            DataQualityMetrics(
                completeness=1.0,
                timeliness=0.0,
                accuracy_score=1.0,
                consistency_score=1.0,
                n_records=-1,
                n_anomalies=0,
            )

    def test_metrics_serialization(self):
        """Test metrics can be serialized."""
        metrics = DataQualityMetrics(
            completeness=0.95,
            timeliness=1.0,
            accuracy_score=0.99,
            consistency_score=0.98,
            n_records=5000,
            n_anomalies=5,
        )
        data = metrics.model_dump()
        assert data["completeness"] == 0.95
        assert data["n_records"] == 5000


class TestDataQualityReport:
    """Tests for DataQualityReport model."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample quality metrics."""
        return DataQualityMetrics(
            completeness=0.98,
            timeliness=1.0,
            accuracy_score=0.99,
            consistency_score=1.0,
            n_records=100000,
            n_anomalies=3,
            n_critical=0,
            n_error=1,
            n_warning=2,
        )

    @pytest.fixture
    def sample_anomaly(self):
        """Create sample anomaly."""
        return DataAnomaly(
            anomaly_type=AnomalyType.PRICE_SPIKE,
            severity=Severity.ERROR,
            timestamp=datetime(2024, 1, 15),
            symbol="AAPL",
            description="Price spike detected",
        )

    def test_create_report(self, sample_metrics):
        """Test creating a quality report."""
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=sample_metrics,
            is_production_ready=True,
        )
        assert report.symbol == "AAPL"
        assert report.source == "databento"
        assert report.frequency == "1min"
        assert report.is_production_ready is True
        assert len(report.anomalies) == 0
        assert len(report.recommendations) == 0

    def test_create_report_with_anomalies(self, sample_metrics, sample_anomaly):
        """Test creating a report with anomalies."""
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=sample_metrics,
            anomalies=[sample_anomaly],
            recommendations=["Review price spikes manually"],
            is_production_ready=False,
        )
        assert len(report.anomalies) == 1
        assert report.anomalies[0].anomaly_type == AnomalyType.PRICE_SPIKE
        assert len(report.recommendations) == 1

    def test_created_at_default(self, sample_metrics):
        """Test created_at defaults to now."""
        before = datetime.utcnow()
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=sample_metrics,
            is_production_ready=True,
        )
        after = datetime.utcnow()
        assert before <= report.created_at <= after

    def test_is_acceptable_default_thresholds(self, sample_metrics):
        """Test is_acceptable with default thresholds."""
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=sample_metrics,
            is_production_ready=True,
        )
        # Default: min_completeness=0.95, max_critical=0, max_errors=5
        # Metrics: completeness=0.98, n_critical=0, n_error=1
        assert report.is_acceptable() is True

    def test_is_acceptable_low_completeness(self):
        """Test is_acceptable with low completeness."""
        metrics = DataQualityMetrics(
            completeness=0.90,  # Below 0.95 threshold
            timeliness=1.0,
            accuracy_score=0.99,
            consistency_score=1.0,
            n_records=100000,
            n_anomalies=0,
        )
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=metrics,
            is_production_ready=False,
        )
        assert report.is_acceptable() is False
        assert report.is_acceptable(min_completeness=0.85) is True

    def test_is_acceptable_critical_anomalies(self):
        """Test is_acceptable with critical anomalies."""
        metrics = DataQualityMetrics(
            completeness=0.99,
            timeliness=1.0,
            accuracy_score=0.99,
            consistency_score=1.0,
            n_records=100000,
            n_anomalies=1,
            n_critical=1,  # Above 0 threshold
        )
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=metrics,
            is_production_ready=False,
        )
        assert report.is_acceptable() is False
        assert report.is_acceptable(max_critical=1) is True

    def test_is_acceptable_error_anomalies(self):
        """Test is_acceptable with too many errors."""
        metrics = DataQualityMetrics(
            completeness=0.99,
            timeliness=1.0,
            accuracy_score=0.99,
            consistency_score=1.0,
            n_records=100000,
            n_anomalies=10,
            n_error=10,  # Above 5 threshold
        )
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=metrics,
            is_production_ready=False,
        )
        assert report.is_acceptable() is False
        assert report.is_acceptable(max_errors=15) is True

    def test_is_acceptable_custom_thresholds(self, sample_metrics):
        """Test is_acceptable with custom thresholds."""
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=sample_metrics,
            is_production_ready=True,
        )
        # Stricter thresholds
        assert report.is_acceptable(min_completeness=0.99) is False
        assert report.is_acceptable(max_errors=0) is False

    def test_summary_basic(self, sample_metrics):
        """Test summary generation."""
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=sample_metrics,
            is_production_ready=True,
        )
        summary = report.summary()
        assert "AAPL" in summary
        assert "databento" in summary
        assert "1min" in summary
        assert "100,000" in summary  # n_records formatted
        assert "98.0%" in summary  # completeness
        assert "YES" in summary  # production ready

    def test_summary_with_recommendations(self, sample_metrics):
        """Test summary includes recommendations."""
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=sample_metrics,
            recommendations=["Check data gaps", "Review outliers"],
            is_production_ready=True,
        )
        summary = report.summary()
        assert "Recommendations" in summary
        assert "Check data gaps" in summary
        assert "Review outliers" in summary

    def test_summary_not_production_ready(self, sample_metrics):
        """Test summary shows NO for not production ready."""
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=sample_metrics,
            is_production_ready=False,
        )
        summary = report.summary()
        assert "NO" in summary

    def test_to_dict(self, sample_metrics, sample_anomaly):
        """Test to_dict serialization."""
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=sample_metrics,
            anomalies=[sample_anomaly],
            recommendations=["Test recommendation"],
            is_production_ready=True,
        )
        data = report.to_dict()

        assert data["symbol"] == "AAPL"
        assert data["source"] == "databento"
        assert data["frequency"] == "1min"
        assert data["is_production_ready"] is True
        assert "metrics" in data
        assert data["metrics"]["completeness"] == 0.98
        assert len(data["anomalies"]) == 1
        assert data["anomalies"][0]["anomaly_type"] == "price_spike"
        assert len(data["recommendations"]) == 1

    def test_to_dict_json_serializable(self, sample_metrics):
        """Test to_dict output is JSON serializable."""
        import json

        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=sample_metrics,
            is_production_ready=True,
        )
        data = report.to_dict()
        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["symbol"] == "AAPL"


class TestDataValidationRequest:
    """Tests for DataValidationRequest model."""

    def test_create_minimal_request(self):
        """Test creating request with minimal fields."""
        request = DataValidationRequest(symbol="AAPL")
        assert request.symbol == "AAPL"
        assert request.date_range is None
        assert len(request.checks) == 4  # default checks
        assert request.include_details is True

    def test_default_checks(self):
        """Test default validation checks."""
        request = DataValidationRequest(symbol="AAPL")
        expected_checks = ["completeness", "stale_data", "price_spikes", "ohlc_validation"]
        assert request.checks == expected_checks

    def test_custom_checks(self):
        """Test custom validation checks."""
        request = DataValidationRequest(
            symbol="AAPL",
            checks=["completeness", "outliers"],
        )
        assert request.checks == ["completeness", "outliers"]

    def test_with_date_range(self):
        """Test request with date range."""
        request = DataValidationRequest(
            symbol="AAPL",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
        )
        assert request.date_range[0] == datetime(2024, 1, 1)
        assert request.date_range[1] == datetime(2024, 6, 30)

    def test_with_thresholds(self):
        """Test request with custom thresholds."""
        request = DataValidationRequest(
            symbol="AAPL",
            thresholds={"price_spike_std": 5.0, "stale_threshold": 10},
        )
        assert request.thresholds["price_spike_std"] == 5.0
        assert request.thresholds["stale_threshold"] == 10

    def test_include_details_false(self):
        """Test request without detailed anomalies."""
        request = DataValidationRequest(
            symbol="AAPL",
            include_details=False,
        )
        assert request.include_details is False

    def test_full_request(self):
        """Test fully specified request."""
        request = DataValidationRequest(
            symbol="AAPL",
            date_range=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
            checks=["completeness", "price_spikes", "ohlc_validation"],
            thresholds={"price_spike_std": 3.0},
            include_details=True,
        )
        assert request.symbol == "AAPL"
        assert len(request.checks) == 3
        assert request.thresholds["price_spike_std"] == 3.0

    def test_request_serialization(self):
        """Test request can be serialized."""
        request = DataValidationRequest(
            symbol="AAPL",
            checks=["completeness"],
            thresholds={"test": 1.0},
        )
        data = request.model_dump()
        assert data["symbol"] == "AAPL"
        assert data["checks"] == ["completeness"]
        assert data["thresholds"] == {"test": 1.0}


class TestIntegrationScenarios:
    """Integration tests for data quality workflow."""

    def test_full_quality_workflow(self):
        """Test complete quality assessment workflow."""
        # 1. Create metrics from data analysis
        metrics = DataQualityMetrics(
            completeness=0.995,
            timeliness=0.5,
            accuracy_score=0.99,
            consistency_score=1.0,
            n_records=100000,
            n_anomalies=3,
            n_critical=0,
            n_error=0,
            n_warning=3,
        )

        # 2. Create anomaly records
        anomalies = [
            DataAnomaly(
                anomaly_type=AnomalyType.ZERO_VOLUME,
                severity=Severity.WARNING,
                timestamp=datetime(2024, 1, 15),
                symbol="AAPL",
                description="Zero volume on trading day",
            ),
            DataAnomaly(
                anomaly_type=AnomalyType.OUTLIER,
                severity=Severity.WARNING,
                timestamp=datetime(2024, 3, 10),
                symbol="AAPL",
                description="Volume 5x above average",
                value=50000000,
                expected_range=(5000000, 15000000),
            ),
        ]

        # 3. Create quality report
        report = DataQualityReport(
            symbol="AAPL",
            source="databento",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=metrics,
            anomalies=anomalies,
            recommendations=["Minor warnings can be ignored"],
            is_production_ready=True,
        )

        # 4. Check acceptability
        assert report.is_acceptable() is True

        # 5. Generate summary
        summary = report.summary()
        assert "AAPL" in summary
        assert "99.5%" in summary

        # 6. Export to dict
        data = report.to_dict()
        assert len(data["anomalies"]) == 2

    def test_validation_request_response_cycle(self):
        """Test request/response pattern."""
        # Create validation request
        request = DataValidationRequest(
            symbol="AAPL",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            checks=["completeness", "price_spikes"],
            thresholds={"price_spike_std": 3.0},
        )

        # Simulate response (in real usage, ml4t-data would generate this)
        metrics = DataQualityMetrics(
            completeness=0.98,
            timeliness=1.0,
            accuracy_score=0.99,
            consistency_score=1.0,
            n_records=50000,
            n_anomalies=0,
        )

        report = DataQualityReport(
            symbol=request.symbol,
            source="databento",
            date_range=request.date_range,
            frequency="1min",
            metrics=metrics,
            is_production_ready=True,
        )

        # Verify response matches request
        assert report.symbol == request.symbol
        assert report.date_range == request.date_range
        assert report.is_acceptable()

    def test_poor_quality_data_handling(self):
        """Test handling of poor quality data."""
        metrics = DataQualityMetrics(
            completeness=0.75,  # Poor
            timeliness=120.0,  # 2 hours stale
            accuracy_score=0.80,
            consistency_score=0.90,
            n_records=50000,
            n_anomalies=25,
            n_critical=2,
            n_error=8,
            n_warning=15,
        )

        report = DataQualityReport(
            symbol="PENNY",
            source="questionable_source",
            date_range=(datetime(2024, 1, 1), datetime(2024, 6, 30)),
            frequency="1min",
            metrics=metrics,
            recommendations=[
                "Data completeness too low for production",
                "Too many critical anomalies",
                "Consider alternative data source",
            ],
            is_production_ready=False,
        )

        # Should fail default acceptability
        assert report.is_acceptable() is False

        # Even with relaxed thresholds, critical anomalies fail
        assert report.is_acceptable(min_completeness=0.70) is False
        # Need to relax all thresholds: completeness, critical, AND errors
        assert report.is_acceptable(min_completeness=0.70, max_critical=2, max_errors=10) is True

        # Summary should show issues
        summary = report.summary()
        assert "NO" in summary  # Not production ready
        assert "75.0%" in summary  # Low completeness
        assert "Recommendations" in summary
