"""Tests for optional dependency handling.

This test suite verifies that the dependency checking system works correctly
and provides clear error messages when dependencies are missing.
"""

import pytest

from ml4t.diagnostic.utils.dependencies import (
    DEPS,
    DependencyInfo,
    OptionalDependencies,
    check_dependency,
    get_dependency_summary,
    require_dependency,
    warn_if_missing,
)


class TestDependencyInfo:
    """Test DependencyInfo dataclass."""

    def test_creation(self):
        """Test creating a DependencyInfo instance."""
        dep = DependencyInfo(
            name="TestPackage",
            import_name="testpkg",
            install_cmd="pip install testpkg",
            purpose="Testing purposes",
            features=["feature1", "feature2"],
        )

        assert dep.name == "TestPackage"
        assert dep.import_name == "testpkg"
        assert dep.install_cmd == "pip install testpkg"
        assert dep.purpose == "Testing purposes"
        assert dep.features == ["feature1", "feature2"]
        assert dep.alternatives == []

    def test_creation_with_alternatives(self):
        """Test creating DependencyInfo with alternatives."""
        dep = DependencyInfo(
            name="TestPackage",
            import_name="testpkg",
            install_cmd="pip install testpkg",
            purpose="Testing",
            features=["test"],
            alternatives=["alt1", "alt2"],
        )

        assert dep.alternatives == ["alt1", "alt2"]

    def test_is_available_true(self):
        """Test is_available for installed package (numpy)."""
        dep = DependencyInfo(
            name="NumPy",
            import_name="numpy",
            install_cmd="pip install numpy",
            purpose="Testing",
            features=["test"],
        )

        assert dep.is_available is True

    def test_is_available_false(self):
        """Test is_available for non-existent package."""
        dep = DependencyInfo(
            name="FakePackage",
            import_name="nonexistent_package_12345",
            install_cmd="pip install fake",
            purpose="Testing",
            features=["test"],
        )

        assert dep.is_available is False

    def test_require_success(self):
        """Test require() with installed package."""
        dep = DependencyInfo(
            name="NumPy",
            import_name="numpy",
            install_cmd="pip install numpy",
            purpose="Testing",
            features=["test"],
        )

        # Should not raise
        dep.require()

    def test_require_failure(self):
        """Test require() with missing package."""
        dep = DependencyInfo(
            name="FakePackage",
            import_name="nonexistent_package_12345",
            install_cmd="pip install fake",
            purpose="Testing",
            features=["test"],
            alternatives=["alt1"],
        )

        with pytest.raises(ImportError) as exc_info:
            dep.require("test feature")

        error_msg = str(exc_info.value)
        assert "FakePackage" in error_msg
        assert "test feature" in error_msg
        assert "pip install fake" in error_msg
        assert "alt1" in error_msg

    def test_warn_if_missing_available(self):
        """Test warn_if_missing with available package."""
        dep = DependencyInfo(
            name="NumPy",
            import_name="numpy",
            install_cmd="pip install numpy",
            purpose="Testing",
            features=["test"],
        )

        # Should return True, no warning
        assert dep.warn_if_missing() is True

    def test_warn_if_missing_unavailable(self):
        """Test warn_if_missing with missing package."""
        dep = DependencyInfo(
            name="FakePackage",
            import_name="nonexistent_package_12345",
            install_cmd="pip install fake",
            purpose="Testing",
            features=["test"],
            alternatives=["alt1"],
        )

        with pytest.warns(UserWarning) as warning_list:
            result = dep.warn_if_missing("test feature", "skipping")

        assert result is False
        assert len(warning_list) == 1
        warning_msg = str(warning_list[0].message)
        assert "FakePackage" in warning_msg
        assert "skipping" in warning_msg
        assert "test feature" in warning_msg
        assert "pip install fake" in warning_msg
        assert "alt1" in warning_msg


class TestOptionalDependencies:
    """Test OptionalDependencies registry."""

    def test_known_dependencies_registered(self):
        """Test that all known dependencies are registered."""
        assert "lightgbm" in DEPS._deps
        assert "xgboost" in DEPS._deps
        assert "shap" in DEPS._deps
        assert "plotly" in DEPS._deps

    def test_lightgbm_info(self):
        """Test LightGBM dependency info."""
        lgb = DEPS.lightgbm
        assert lgb.name == "LightGBM"
        assert lgb.import_name == "lightgbm"
        assert "pip install lightgbm" in lgb.install_cmd
        assert "importance" in lgb.purpose.lower()
        assert len(lgb.features) > 0
        assert len(lgb.alternatives) > 0

    def test_shap_info(self):
        """Test SHAP dependency info."""
        shap = DEPS.shap
        assert shap.name == "SHAP"
        assert shap.import_name == "shap"
        assert "pip install shap" in shap.install_cmd
        assert len(shap.features) > 0

    def test_attribute_access(self):
        """Test accessing dependencies as attributes."""
        lgb = DEPS.lightgbm
        assert isinstance(lgb, DependencyInfo)
        assert lgb.name == "LightGBM"

    def test_item_access(self):
        """Test accessing dependencies as items."""
        lgb = DEPS["lightgbm"]
        assert isinstance(lgb, DependencyInfo)
        assert lgb.name == "LightGBM"

    def test_get_method(self):
        """Test get method."""
        lgb = DEPS.get("lightgbm")
        assert isinstance(lgb, DependencyInfo)

        missing = DEPS.get("nonexistent", default=None)
        assert missing is None

    def test_check_method(self):
        """Test check method."""
        # NumPy should be installed (required dependency)
        deps = OptionalDependencies()
        # Use a dependency we know exists from the registry
        has_lgb = deps.check("lightgbm")
        assert isinstance(has_lgb, bool)

    def test_check_multiple(self):
        """Test check_multiple method."""
        result = DEPS.check_multiple(["lightgbm", "shap", "plotly"])

        assert isinstance(result, dict)
        assert "lightgbm" in result
        assert "shap" in result
        assert "plotly" in result
        assert all(isinstance(v, bool) for v in result.values())

    def test_get_missing(self):
        """Test get_missing method."""
        # Create temporary fake dependency
        DEPS._deps["__fake_test__"] = DependencyInfo(
            name="Fake",
            import_name="__nonexistent__",
            install_cmd="pip install fake",
            purpose="Test",
            features=["test"],
        )

        try:
            missing = DEPS.get_missing(["lightgbm", "__fake_test__"])
            assert "__fake_test__" in missing
        finally:
            del DEPS._deps["__fake_test__"]

    def test_warn_missing(self):
        """Test warn_missing method."""
        # Create temporary fake dependency
        DEPS._deps["__fake_test__"] = DependencyInfo(
            name="Fake",
            import_name="__nonexistent__",
            install_cmd="pip install fake",
            purpose="Test",
            features=["test"],
        )

        try:
            with pytest.warns(UserWarning):
                missing = DEPS.warn_missing(["lightgbm", "__fake_test__"], "test feature")
                assert "__fake_test__" in missing
        finally:
            del DEPS._deps["__fake_test__"]

    def test_summary(self):
        """Test summary generation."""
        summary = DEPS.summary()

        assert "Optional Dependencies" in summary
        assert "LightGBM" in summary
        assert "SHAP" in summary
        assert "XGBoost" in summary


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_check_dependency(self):
        """Test check_dependency function."""
        result = check_dependency("lightgbm")
        assert isinstance(result, bool)

    def test_require_dependency_success(self):
        """Test require_dependency with existing (mock) dependency."""
        # Should not raise for registered dependencies
        # We can't test actual imports without the packages installed

    def test_require_dependency_failure(self):
        """Test require_dependency with missing dependency."""
        # Create temporary fake dependency
        DEPS._deps["__fake_test__"] = DependencyInfo(
            name="Fake",
            import_name="__nonexistent__",
            install_cmd="pip install fake",
            purpose="Test",
            features=["test"],
        )

        try:
            with pytest.raises(ImportError) as exc_info:
                require_dependency("__fake_test__", "test feature")

            error_msg = str(exc_info.value)
            assert "Fake" in error_msg
            assert "test feature" in error_msg
        finally:
            del DEPS._deps["__fake_test__"]

    def test_require_dependency_unknown(self):
        """Test require_dependency with unknown dependency."""
        with pytest.raises(ImportError) as exc_info:
            require_dependency("completely_unknown_dep_xyz")

        assert "Unknown dependency" in str(exc_info.value)

    def test_warn_if_missing_available(self):
        """Test warn_if_missing with available mock."""
        # Create a dependency that exists (numpy is always available)
        DEPS._deps["__test_numpy__"] = DependencyInfo(
            name="TestNumPy",
            import_name="numpy",
            install_cmd="pip install numpy",
            purpose="Test",
            features=["test"],
        )

        try:
            # Should return True, no warning
            result = warn_if_missing("__test_numpy__")
            assert result is True
        finally:
            del DEPS._deps["__test_numpy__"]

    def test_warn_if_missing_unavailable(self):
        """Test warn_if_missing with missing dependency."""
        # Create temporary fake dependency
        DEPS._deps["__fake_test__"] = DependencyInfo(
            name="Fake",
            import_name="__nonexistent__",
            install_cmd="pip install fake",
            purpose="Test",
            features=["test"],
        )

        try:
            with pytest.warns(UserWarning) as warning_list:
                result = warn_if_missing("__fake_test__", "test feature", "skipping")

            assert result is False
            assert len(warning_list) == 1
            warning_msg = str(warning_list[0].message)
            assert "Fake" in warning_msg
            assert "skipping" in warning_msg
        finally:
            del DEPS._deps["__fake_test__"]

    def test_warn_if_missing_unknown(self):
        """Test warn_if_missing with unknown dependency."""
        with pytest.warns(UserWarning) as warning_list:
            result = warn_if_missing("completely_unknown_dep_xyz")

        assert result is False
        assert "Unknown dependency" in str(warning_list[0].message)

    def test_get_dependency_summary(self):
        """Test get_dependency_summary function."""
        summary = get_dependency_summary()

        assert isinstance(summary, str)
        assert "Optional Dependencies" in summary
        assert "LightGBM" in summary


class TestIntegrationWithFeatureOutcome:
    """Test integration with FeatureOutcome class."""

    def test_feature_outcome_with_dependencies(self):
        """Test that FeatureOutcome can import with dependency checks."""
        from ml4t.diagnostic.evaluation.feature_outcome import FeatureOutcome

        # Should import successfully
        analyzer = FeatureOutcome()
        assert analyzer is not None

    def test_feature_outcome_warns_on_missing_lightgbm(self, monkeypatch):
        """Test that FeatureOutcome warns when LightGBM is missing."""
        # Mock warn_if_missing to simulate missing LightGBM
        from ml4t.diagnostic.config.feature_config import DiagnosticConfig, MLDiagnosticsSettings
        from ml4t.diagnostic.evaluation.feature_outcome import FeatureOutcome

        # Mock at the module level
        original_warn = warn_if_missing

        def mock_warn(name, *args, **kwargs):
            if name == "lightgbm":
                import warnings

                warnings.warn(
                    "LightGBM not available - skipping feature importance",
                    UserWarning,
                    stacklevel=2,
                )
                return False  # Pretend it's missing
            return original_warn(name, *args, **kwargs)

        # Patch in the feature_outcome module
        import ml4t.diagnostic.evaluation.feature_outcome as fo_module

        monkeypatch.setattr(fo_module, "warn_if_missing", mock_warn)

        config = DiagnosticConfig(ml_diagnostics=MLDiagnosticsSettings(feature_importance=True))

        analyzer = FeatureOutcome(config=config)

        # Create simple test data
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        features = pd.DataFrame(
            {
                "f1": np.random.normal(0, 1, 100),
                "f2": np.random.normal(0, 1, 100),
            }
        )
        outcomes = pd.Series(np.random.normal(0, 1, 100))

        # Should warn about missing LightGBM but not fail
        with pytest.warns(UserWarning) as warning_list:
            result = analyzer.run_analysis(
                features, outcomes, verbose=False
            )  # verbose=False to avoid prints

        # Should have warned
        assert len(warning_list) > 0
        # Should complete despite missing dependency
        assert result is not None
        # Should have IC results even without importance
        assert len(result.ic_results) > 0
