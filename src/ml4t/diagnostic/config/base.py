"""Base configuration classes and shared utilities.

This module provides foundation classes used throughout the config system:
- BaseConfig: Serialization, validation utilities, comparison
- StatisticalTestConfig: Base for all statistical tests
- RuntimeConfig: Execution settings (n_jobs, caching, verbosity)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class BaseConfig(BaseModel):
    """Base configuration class with serialization and comparison utilities.

    All ML4T Diagnostic configs inherit from this class to get consistent behavior
    for serialization, validation, and comparison.

    Examples:
        >>> class MyConfig(BaseConfig):
        ...     value: int = 42
        >>> config = MyConfig()
        >>> config.to_yaml("config.yaml")
        >>> loaded = MyConfig.from_yaml("config.yaml")
        >>> assert config == loaded
    """

    model_config = ConfigDict(
        extra="forbid",  # Catch typos in field names
        validate_assignment=True,  # Validate on attribute assignment
        arbitrary_types_allowed=True,  # Allow Path, etc.
        use_enum_values=True,  # Serialize enums as values
    )

    def to_dict(self, *, exclude_none: bool = False, mode: str = "python") -> dict[str, Any]:
        """Convert config to dictionary.

        Args:
            exclude_none: Exclude fields with None values
            mode: "python" for Python objects, "json" for JSON-serializable

        Returns:
            Dictionary representation of config
        """
        return self.model_dump(exclude_none=exclude_none, mode=mode)

    def to_json(self, file_path: str | Path, *, indent: int = 2) -> None:
        """Save config to JSON file.

        Args:
            file_path: Output file path
            indent: JSON indentation (default 2)
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(self.to_dict(mode="json"), f, indent=indent)

    @classmethod
    def from_json(cls, file_path: str | Path) -> BaseConfig:
        """Load config from JSON file.

        Args:
            file_path: Input file path

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        path = Path(file_path)
        with path.open() as f:
            data = json.load(f)
        return cls(**data)

    def to_yaml(self, file_path: str | Path) -> None:
        """Save config to YAML file.

        Args:
            file_path: Output file path
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            yaml.dump(self.to_dict(mode="json"), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> BaseConfig:
        """Load config from YAML file.

        Args:
            file_path: Input file path

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If YAML is invalid
        """
        path = Path(file_path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseConfig:
        """Load config from dictionary with validation.

        Args:
            data: Dictionary representation of config

        Returns:
            Config instance

        Raises:
            ValidationError: If data is invalid

        Examples:
            >>> config = MyConfig.from_dict({"value": 42})
            >>> assert config.value == 42
        """
        return cls(**data)

    @classmethod
    def from_file(cls, file_path: str | Path) -> BaseConfig:
        """Auto-detect file type (YAML/JSON) and load config.

        Detects file type based on extension (.yaml, .yml, .json).

        Args:
            file_path: Input file path

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is unsupported or content is invalid

        Examples:
            >>> config = MyConfig.from_file("config.yaml")
            >>> config = MyConfig.from_file("config.json")
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif suffix == ".json":
            return cls.from_json(path)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. Supported types: .yaml, .yml, .json"
            )

    def validate_fully(self) -> list[str]:
        """Run all validators and return list of validation issues.

        This method re-validates the config and collects any validation errors
        or warnings. Useful for checking configuration validity after modifications.

        Returns:
            List of validation error messages (empty if valid)

        Examples:
            >>> config = MyConfig(value=42)
            >>> errors = config.validate_fully()
            >>> if errors:
            ...     print(f"Validation errors: {errors}")
        """
        try:
            # Trigger validation by creating a new instance with same data
            self.model_validate(self.model_dump())
            return []
        except ValidationError as e:
            # Parse validation errors from Pydantic
            errors = []
            for error in e.errors():
                loc = ".".join(str(x) for x in error["loc"])
                msg = error["msg"]
                errors.append(f"{loc}: {msg}")
            return errors
        except Exception as e:
            return [str(e)]

    def diff(self, other: BaseConfig) -> dict[str, tuple[Any, Any]]:
        """Compare this config with another, returning differences.

        Args:
            other: Config to compare against

        Returns:
            Dictionary mapping field paths to (self_value, other_value) tuples

        Examples:
            >>> config1 = MyConfig(value=42)
            >>> config2 = MyConfig(value=100)
            >>> config1.diff(config2)
            {'value': (42, 100)}
        """
        if type(self) is not type(other):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}")

        differences = {}
        self_dict = self.to_dict()
        other_dict = other.to_dict()

        def _compare_nested(d1: dict, d2: dict, prefix: str = "") -> None:
            """Recursively compare nested dictionaries."""
            all_keys = set(d1.keys()) | set(d2.keys())
            for key in all_keys:
                path = f"{prefix}.{key}" if prefix else key
                v1, v2 = d1.get(key), d2.get(key)

                if v1 != v2:
                    if isinstance(v1, dict) and isinstance(v2, dict):
                        _compare_nested(v1, v2, path)
                    else:
                        differences[path] = (v1, v2)

        _compare_nested(self_dict, other_dict)
        return differences


class StatisticalTestConfig(BaseConfig):
    """Base configuration for statistical tests.

    Provides common fields for hypothesis tests (significance level, etc.)
    that are inherited by specific test configs.

    Attributes:
        enabled: Whether to run this test
        significance_level: Significance level for hypothesis test (0.01, 0.05, or 0.10)
    """

    enabled: bool = Field(True, description="Whether to run this test")
    significance_level: float = Field(
        0.05,
        ge=0.001,
        le=0.10,
        description="Significance level for hypothesis tests (common: 0.01, 0.05, 0.10)",
    )


class RuntimeConfig(BaseConfig):
    """Configuration for execution settings.

    Centralizes computational resources, caching, and randomness across all
    evaluation functions. Pass as a separate parameter to analysis functions.

    Attributes:
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for serial)
        cache_enabled: Enable caching of expensive computations
        cache_dir: Directory for cache storage
        cache_ttl: Cache time-to-live in seconds (None for no expiration)
        verbose: Enable verbose output
        random_state: Random seed for reproducibility

    Examples:
        >>> from ml4t.diagnostic.config import RuntimeConfig, DiagnosticConfig
        >>> runtime = RuntimeConfig(n_jobs=4, verbose=True)
        >>> result = analyze_features(df, config=DiagnosticConfig(), runtime=runtime)
    """

    n_jobs: int = Field(
        -1,
        ge=-1,
        description="Number of parallel jobs (-1 for all cores, 1 for serial)",
    )
    cache_enabled: bool = Field(True, description="Enable caching of expensive computations")
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "ml4t-diagnostic",
        description="Directory for cache storage",
    )
    cache_ttl: int | None = Field(
        None,
        ge=0,
        description="Cache time-to-live in seconds (None for no expiration)",
    )
    verbose: bool = Field(False, description="Enable verbose output")
    random_state: int | None = Field(None, ge=0, description="Random seed for reproducibility")

    def model_post_init(self, __context: Any) -> None:
        """Create cache directory if it doesn't exist."""
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
