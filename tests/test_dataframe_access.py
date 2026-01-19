"""Tests for DataFrame access API (FR-I4).

This module tests the programmatic DataFrame access functionality for QEngine
integration and further analysis workflows.
"""

import polars as pl
import pytest

from ml4t.diagnostic.results.base import BaseResult
from ml4t.diagnostic.results.feature_results import ACFResult, StationarityTestResult


class TestBaseResultDataFrameAPI:
    """Test BaseResult DataFrame access methods."""

    def test_base_result_get_dataframe_not_implemented(self):
        """Test that BaseResult.get_dataframe() raises NotImplementedError."""

        # Create minimal subclass that doesn't override get_dataframe
        class MinimalResult(BaseResult):
            analysis_type: str = "test"

            def summary(self) -> str:
                return "Test"

        result = MinimalResult(analysis_type="test")

        with pytest.raises(NotImplementedError, match="must implement get_dataframe"):
            result.get_dataframe()

    def test_base_result_list_available_dataframes_default(self):
        """Test that default list_available_dataframes() returns ['primary']."""

        class MinimalResult(BaseResult):
            analysis_type: str = "test"

            def summary(self) -> str:
                return "Test"

            def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
                return pl.DataFrame({"test": [1, 2, 3]})

        result = MinimalResult(analysis_type="test")
        assert result.list_available_dataframes() == ["primary"]

    def test_get_dataframe_schema(self):
        """Test get_dataframe_schema() returns correct column types."""

        class TestResult(BaseResult):
            analysis_type: str = "test"

            def summary(self) -> str:
                return "Test"

            def get_dataframe(self, name: str | None = None) -> pl.DataFrame:
                return pl.DataFrame(
                    {
                        "feature": ["a", "b"],
                        "value": [1.5, 2.5],
                        "is_valid": [True, False],
                    }
                )

        result = TestResult(analysis_type="test")
        schema = result.get_dataframe_schema()

        assert isinstance(schema, dict)
        assert "feature" in schema
        assert "value" in schema
        assert "is_valid" in schema
        assert "String" in schema["feature"] or "Utf8" in schema["feature"]
        assert "Float64" in schema["value"]
        assert "Boolean" in schema["is_valid"]


class TestStationarityTestResultDataFrameAccess:
    """Test StationarityTestResult DataFrame access."""

    def test_list_available_dataframes(self):
        """Test list_available_dataframes() returns correct list."""
        result = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
        )
        available = result.list_available_dataframes()
        assert available == ["primary"]

    def test_get_dataframe_default(self):
        """Test get_dataframe() returns primary DataFrame."""
        result = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
            kpss_statistic=0.2,
            kpss_pvalue=0.10,
            kpss_is_stationary=True,
        )
        df = result.get_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "feature" in df.columns
        assert "adf_statistic" in df.columns
        assert "adf_pvalue" in df.columns
        assert "adf_stationary" in df.columns
        assert "kpss_statistic" in df.columns
        assert len(df) == 1
        assert df["feature"][0] == "returns"
        assert df["adf_pvalue"][0] == 0.01

    def test_get_dataframe_with_primary_name(self):
        """Test get_dataframe('primary') returns same as default."""
        result = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
        )
        df_default = result.get_dataframe()
        df_primary = result.get_dataframe("primary")

        assert df_default.equals(df_primary)

    def test_get_dataframe_invalid_name(self):
        """Test get_dataframe() raises ValueError for invalid name."""
        result = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
        )

        with pytest.raises(ValueError, match="Unknown DataFrame name"):
            result.get_dataframe("nonexistent")

    def test_get_dataframe_schema(self):
        """Test get_dataframe_schema() returns correct types."""
        result = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
        )
        schema = result.get_dataframe_schema()

        assert isinstance(schema, dict)
        assert "feature" in schema
        assert "adf_statistic" in schema
        assert "adf_pvalue" in schema
        assert "adf_stationary" in schema

    def test_dataframe_handles_none_values(self):
        """Test DataFrame access handles None values correctly."""
        result = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
            # kpss fields are None
        )
        df = result.get_dataframe()

        assert df["kpss_statistic"][0] is None
        assert df["kpss_pvalue"][0] is None
        assert df["kpss_stationary"][0] is None


class TestACFResultDataFrameAccess:
    """Test ACFResult DataFrame access."""

    def test_list_available_dataframes(self):
        """Test list_available_dataframes() returns correct list."""
        result = ACFResult(
            feature_name="returns",
            acf_values=[1.0, 0.5, 0.25],
            pacf_values=[1.0, 0.4, 0.1],
        )
        available = result.list_available_dataframes()
        assert available == ["primary"]

    def test_get_dataframe_default(self):
        """Test get_dataframe() returns lag/ACF/PACF DataFrame."""
        result = ACFResult(
            feature_name="returns",
            acf_values=[1.0, 0.5, 0.25, 0.1],
            pacf_values=[1.0, 0.4, 0.1, 0.05],
            significant_lags_acf=[1, 2],
            significant_lags_pacf=[1],
        )
        df = result.get_dataframe()

        assert isinstance(df, pl.DataFrame)
        assert "lag" in df.columns
        assert "acf" in df.columns
        assert "pacf" in df.columns
        assert len(df) == 4  # 4 lags
        assert df["lag"].to_list() == [0, 1, 2, 3]
        assert df["acf"].to_list() == [1.0, 0.5, 0.25, 0.1]
        assert df["pacf"].to_list() == [1.0, 0.4, 0.1, 0.05]

    def test_get_dataframe_with_primary_name(self):
        """Test get_dataframe('primary') returns same as default."""
        result = ACFResult(
            feature_name="returns",
            acf_values=[1.0, 0.5, 0.25],
            pacf_values=[1.0, 0.4, 0.1],
        )
        df_default = result.get_dataframe()
        df_primary = result.get_dataframe("primary")

        assert df_default.equals(df_primary)

    def test_get_dataframe_invalid_name(self):
        """Test get_dataframe() raises ValueError for invalid name."""
        result = ACFResult(
            feature_name="returns",
            acf_values=[1.0, 0.5],
            pacf_values=[1.0, 0.4],
        )

        with pytest.raises(ValueError, match="Unknown DataFrame name"):
            result.get_dataframe("nonexistent")

    def test_get_dataframe_schema(self):
        """Test get_dataframe_schema() returns correct types."""
        result = ACFResult(
            feature_name="returns",
            acf_values=[1.0, 0.5, 0.25],
            pacf_values=[1.0, 0.4, 0.1],
        )
        schema = result.get_dataframe_schema()

        assert isinstance(schema, dict)
        assert "lag" in schema
        assert "acf" in schema
        assert "pacf" in schema


class TestQEngineIntegration:
    """Test DataFrame access for QEngine integration workflows."""

    def test_qengine_storage_workflow(self):
        """Test complete workflow for QEngine storage.

        Simulates how QEngine would:
        1. List available DataFrames
        2. Get DataFrame schema
        3. Retrieve DataFrame data
        4. Store in database
        """
        # Create result
        result = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
        )

        # 1. Discover available DataFrames
        available = result.list_available_dataframes()
        assert len(available) > 0

        # 2. Get schema for each DataFrame
        for name in available:
            schema = result.get_dataframe_schema(name)
            assert isinstance(schema, dict)
            assert len(schema) > 0

        # 3. Retrieve DataFrame
        df = result.get_dataframe(available[0])
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

        # 4. Convert to dict for storage (simulated)
        storage_dict = df.to_dicts()
        assert isinstance(storage_dict, list)
        assert len(storage_dict) == len(df)

    def test_multiple_results_aggregation(self):
        """Test aggregating DataFrames from multiple results.

        Simulates how QEngine would aggregate results from multiple
        features for batch storage.
        """
        # Create multiple results
        results = [
            StationarityTestResult(
                feature_name=f"feature_{i}",
                adf_statistic=-3.5 - i * 0.1,
                adf_pvalue=0.01 + i * 0.01,
                adf_is_stationary=True,
            )
            for i in range(3)
        ]

        # Aggregate DataFrames
        dfs = [r.get_dataframe() for r in results]
        combined = pl.concat(dfs)

        assert len(combined) == 3
        assert "feature" in combined.columns
        assert combined["feature"].to_list() == ["feature_0", "feature_1", "feature_2"]

    def test_json_and_dataframe_export(self):
        """Test that both JSON and DataFrame exports work together.

        QEngine may want both formats:
        - JSON for metadata storage
        - DataFrame for time-series storage
        """
        result = ACFResult(
            feature_name="returns",
            acf_values=[1.0, 0.5, 0.25],
            pacf_values=[1.0, 0.4, 0.1],
        )

        # JSON export
        json_str = result.to_json_string()
        assert isinstance(json_str, str)
        assert "acf_values" in json_str

        # DataFrame export
        df = result.get_dataframe()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3

        # Both should contain the same data
        assert df["acf"].to_list() == result.acf_values
        assert df["pacf"].to_list() == result.pacf_values

    def test_schema_discovery_before_data_load(self):
        """Test schema discovery without loading full DataFrame.

        Useful for QEngine to prepare storage schema before loading data.
        """
        result = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
        )

        # Get schema first
        schema = result.get_dataframe_schema()
        expected_columns = {
            "feature",
            "adf_statistic",
            "adf_pvalue",
            "adf_stationary",
            "adf_lags_used",
            "adf_n_obs",
            "kpss_statistic",
            "kpss_pvalue",
            "kpss_stationary",
            "pp_statistic",
            "pp_pvalue",
            "pp_stationary",
        }
        assert set(schema.keys()) == expected_columns

        # Then load data
        df = result.get_dataframe()
        assert set(df.columns) == expected_columns


class TestDataFrameAccessDocumentation:
    """Test that DataFrame access is well-documented and discoverable."""

    def test_list_available_dataframes_is_discoverable(self):
        """Test that users can discover available DataFrames."""
        result = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
        )

        # Method exists
        assert hasattr(result, "list_available_dataframes")
        assert callable(result.list_available_dataframes)

        # Returns sensible data
        available = result.list_available_dataframes()
        assert isinstance(available, list)
        assert all(isinstance(name, str) for name in available)

    def test_get_dataframe_has_docstring(self):
        """Test that get_dataframe() has helpful documentation."""
        result = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
        )

        doc = result.get_dataframe.__doc__
        assert doc is not None
        assert "DataFrame" in doc or "dataframe" in doc.lower()

    def test_error_messages_are_helpful(self):
        """Test that error messages guide users to correct usage."""
        result = StationarityTestResult(
            feature_name="returns",
            adf_statistic=-3.5,
            adf_pvalue=0.01,
            adf_is_stationary=True,
        )

        try:
            result.get_dataframe("invalid_name")
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            # Error should mention what's available
            assert "Available" in error_msg or "available" in error_msg
            # Should show the valid options
            assert "primary" in error_msg
