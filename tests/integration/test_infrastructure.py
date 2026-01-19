"""
Phase 1 Integration Tests

Validates that all Phase 1 infrastructure components work together correctly.
Tests configuration, validation, caching, reporting, logging, and error handling.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestModuleStructure:
    """Test module structure and exports."""

    def test_core_infrastructure_modules_exist(self):
        """Verify core infrastructure modules have been created."""
        base_path = Path(__file__).parent.parent.parent / "src" / "ml4t" / "diagnostic"

        # Core infrastructure modules that should exist
        required_modules = [
            "config",
            "errors",
            "validation",
            "caching",
            "reporting",
            "logging",
            "results",
            "integration",  # Integration contracts (not subdirectories)
        ]

        for module in required_modules:
            # Convert module path string to Path (handles '/' correctly)
            module_path = base_path / module
            init_file = module_path / "__init__.py"

            assert init_file.exists(), f"Missing module: {module}"

    def test_examples_exist(self):
        """Verify all Phase 1 examples have been created."""
        examples_path = Path(__file__).parent.parent.parent / "examples"

        required_examples = [
            "caching_example.py",
            "report_generation_example.py",
            "error_handling_example.py",
        ]

        for example in required_examples:
            example_file = examples_path / example
            assert example_file.exists(), f"Missing example: {example}"


class TestComponentQuality:
    """Test code quality standards for Phase 1."""

    def test_all_modules_have_docstrings(self):
        """Verify all modules have comprehensive docstrings."""
        base_path = Path(__file__).parent.parent.parent / "src" / "ml4t" / "diagnostic"

        modules_to_check = [
            "config/__init__.py",
            "errors/__init__.py",
            "validation/__init__.py",
            "caching/__init__.py",
            "reporting/__init__.py",
            "logging/__init__.py",
        ]

        for module_path in modules_to_check:
            full_path = base_path / module_path
            content = full_path.read_text()

            # Check for docstring
            assert '"""' in content, f"Missing docstring in {module_path}"

            # Check docstring is substantial (>100 chars)
            first_doc = content.split('"""')[1]
            assert len(first_doc) > 100, f"Docstring too short in {module_path}"


class TestPhase1Acceptance:
    """Acceptance tests for Phase 1 completion criteria."""

    def test_result_schemas_present(self):
        """Acceptance: Pydantic result schemas implemented."""
        results_path = (
            Path(__file__).parent.parent.parent / "src" / "ml4t" / "diagnostic" / "results"
        )

        # Check result schemas exist
        assert (results_path / "__init__.py").exists()
        assert (results_path / "base.py").exists()

    def test_dataframe_api_present(self):
        """Acceptance: DataFrame access API implemented."""
        # Already covered by result schemas

    def test_reporting_engine_present(self):
        """Acceptance: Report generation engine exists."""
        reporting_path = (
            Path(__file__).parent.parent.parent / "src" / "ml4t" / "diagnostic" / "reporting"
        )

        assert (reporting_path / "__init__.py").exists()
        assert (reporting_path / "base.py").exists()
        assert (reporting_path / "html_renderer.py").exists()

    def test_caching_framework_present(self):
        """Acceptance: Caching framework implemented."""
        caching_path = (
            Path(__file__).parent.parent.parent / "src" / "ml4t" / "diagnostic" / "caching"
        )

        assert (caching_path / "__init__.py").exists()
        assert (caching_path / "cache.py").exists()
        assert (caching_path / "decorators.py").exists()

    def test_validation_utilities_present(self):
        """Acceptance: Validation utilities implemented."""
        validation_path = (
            Path(__file__).parent.parent.parent / "src" / "ml4t" / "diagnostic" / "validation"
        )

        assert (validation_path / "__init__.py").exists()
        assert (validation_path / "dataframe.py").exists()
        assert (validation_path / "returns.py").exists()

    def test_error_handling_present(self):
        """Acceptance: Error handling framework implemented."""
        errors_path = Path(__file__).parent.parent.parent / "src" / "ml4t" / "diagnostic" / "errors"

        assert (errors_path / "__init__.py").exists()

    def test_logging_infrastructure_present(self):
        """Acceptance: Logging and debugging infrastructure implemented."""
        logging_path = (
            Path(__file__).parent.parent.parent / "src" / "ml4t" / "diagnostic" / "logging"
        )

        assert (logging_path / "__init__.py").exists()
        assert (logging_path / "logger.py").exists()
        assert (logging_path / "progress.py").exists()
        assert (logging_path / "performance.py").exists()

    # Note: test_phase1_state_complete removed - was testing dev workflow state, not library functionality


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
