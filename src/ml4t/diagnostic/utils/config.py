"""Legacy configuration management for ml4t-diagnostic evaluation workflows.

This module provides the original YAML-based evaluator configuration manager.
New code should prefer the canonical configuration schemas under
`ml4t.diagnostic.config`.
"""

import os
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigError(Exception):
    """Raised when configuration is invalid."""


class SplitterConfig(BaseModel):
    """Configuration schema for cross-validation splitters."""

    type: Literal["WalkForwardCV", "CombinatorialCV"] = Field(
        description="Type of cross-validation splitter",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the splitter",
    )

    @field_validator("params")
    @classmethod
    def validate_splitter_params(cls, v: dict[str, Any], info) -> dict[str, Any]:
        """Validate splitter-specific parameters."""
        splitter_type = info.data.get("type")

        if splitter_type == "WalkForwardCV":
            # Validate walk-forward specific parameters
            if "n_splits" in v and (v["n_splits"] < 2 or v["n_splits"] > 50):
                raise ValueError("n_splits must be between 2 and 50")
            if "test_size" in v and (v["test_size"] <= 0 or v["test_size"] >= 1):
                raise ValueError("test_size must be between 0 and 1")

        elif splitter_type == "CombinatorialCV":
            # Validate combinatorial specific parameters
            if "n_groups" in v and (v["n_groups"] < 2 or v["n_groups"] > 20):
                raise ValueError("n_groups must be between 2 and 20")

        return v


class DataConfig(BaseModel):
    """Configuration schema for data handling parameters."""

    label_horizon: int = Field(
        ge=0,
        le=252,
        default=20,
        description="Forward-looking period of labels (in periods)",
    )
    embargo_pct: float = Field(
        ge=0.0,
        le=1.0,
        default=0.01,
        description="Embargo percentage to prevent leakage",
    )
    min_samples_per_fold: int = Field(
        ge=10,
        le=10000,
        default=100,
        description="Minimum number of samples required per fold",
    )


class VisualizationConfig(BaseModel):
    """Configuration schema for visualization settings."""

    theme: Literal["default", "dark", "light"] = Field(
        default="default",
        description="Visualization theme",
    )
    export_format: Literal["html", "png", "pdf", "svg"] = Field(
        default="html",
        description="Export format for visualizations",
    )
    include_dashboard: bool = Field(
        default=True,
        description="Whether to include interactive dashboard",
    )


class LoggingConfig(BaseModel):
    """Configuration schema for logging settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    use_wandb: bool = Field(
        default=False,
        description="Whether to use Weights & Biases logging",
    )
    wandb_project: str | None = Field(default=None, description="W&B project name")
    wandb_entity: str | None = Field(default=None, description="W&B entity name")

    @model_validator(mode="after")
    def validate_wandb_config(self):
        """Validate W&B configuration consistency."""
        if self.use_wandb and not self.wandb_project:
            raise ValueError("wandb_project is required when use_wandb=True")
        return self


class EvaluatorConfig(BaseModel):
    """Configuration schema for the main Evaluator class."""

    tier: int = Field(
        ge=1,
        le=3,
        default=2,
        description="Validation tier level (1=rigorous, 2=standard, 3=fast)",
    )
    confidence_level: float = Field(
        gt=0.0,
        lt=1.0,
        default=0.05,
        description="Significance level for statistical tests",
    )
    bootstrap_samples: int = Field(
        ge=100,
        le=10000,
        default=1000,
        description="Number of bootstrap samples for confidence intervals",
    )
    random_state: int | None = Field(
        ge=0,
        le=2**31 - 1,
        default=None,
        description="Random seed for reproducible results",
    )
    n_jobs: int = Field(
        ge=-1,
        le=128,
        default=1,
        description="Number of parallel jobs (-1 for all cores)",
    )


class QEvalConfig(BaseModel):
    """Complete configuration schema for ml4t-diagnostic evaluation workflows."""

    evaluation: EvaluatorConfig = Field(
        default_factory=EvaluatorConfig,
        description="Main evaluator configuration",
    )
    splitter: SplitterConfig = Field(
        description="Cross-validation splitter configuration",
    )
    metrics: list[Literal["ic", "sharpe", "sortino", "max_drawdown", "hit_rate"]] = Field(
        default=["ic", "sharpe", "hit_rate"],
        min_length=1,
        max_length=10,
        description="List of metrics to compute",
    )
    statistical_tests: dict[Literal["tier_1", "tier_2", "tier_3"], list[str]] = Field(
        default={"tier_1": ["dsr", "fdr"], "tier_2": ["hac_ic"], "tier_3": []},
        description="Statistical tests by tier",
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data handling configuration",
    )
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Visualization settings",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration",
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics_non_empty(cls, v: list[str]) -> list[str]:
        """Ensure at least one metric is specified."""
        if not v:
            raise ValueError("At least one metric must be specified")
        return v

    @model_validator(mode="after")
    def validate_tier_consistency(self):
        """Validate configuration consistency across tiers."""
        tier = self.evaluation.tier

        # Tier 1 should use CombinatorialCV for maximum rigor
        if tier == 1 and self.splitter.type != "CombinatorialCV":
            raise ValueError(
                "Tier 1 evaluation should use CombinatorialCV for maximum rigor",
            )

        # Tier 3 should have minimal statistical tests
        if tier == 3 and len(self.statistical_tests.get("tier_3", [])) > 2:
            raise ValueError(
                "Tier 3 is designed for fast screening - limit statistical tests",
            )

        return self


# Explicit legacy alias to distinguish from `ml4t.diagnostic.config` models.
LegacyEvaluationConfig = QEvalConfig


class EvaluationConfigManager:
    """Enhanced configuration manager with Pydantic validation.

    This class loads and validates YAML configuration files
    for ml4t-diagnostic evaluation pipelines using Pydantic schemas.
    """

    def __init__(self, config_path: str | Path | None = None):
        """Initialize configuration manager.

        Parameters
        ----------
        config_path : str or Path, optional
            Path to YAML configuration file. If None, uses defaults.
        """
        # Start with default configuration
        default_config = self._create_default_config()

        if config_path is not None:
            # Load and merge user configuration
            user_config = self._load_from_yaml(config_path)
            self.config = self._merge_configs(default_config, user_config)
        else:
            self.config = default_config

    def _create_default_config(self) -> QEvalConfig:
        """Create default configuration with all required fields."""
        return QEvalConfig(
            splitter=SplitterConfig(
                type="WalkForwardCV",
                params={
                    "n_splits": 5,
                    "test_size": 0.2,
                    "gap": 0,
                    "expanding": True,
                },
            ),
        )

    def _load_from_yaml(self, config_path: str | Path) -> dict[str, Any]:
        """Load configuration from YAML file with validation.

        Parameters
        ----------
        config_path : str or Path
            Path to YAML configuration file

        Returns:
        -------
        dict
            Raw configuration dictionary

        Raises:
        ------
        ConfigError
            If file cannot be loaded or contains invalid YAML
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {config_path}: {e}") from e

        if user_config is None:
            user_config = {}

        return user_config

    def _merge_configs(
        self,
        base_config: QEvalConfig,
        user_config: dict[str, Any],
    ) -> QEvalConfig:
        """Merge user configuration with base configuration using Pydantic validation.

        Parameters
        ----------
        base_config : QEvalConfig
            Base configuration schema
        user_config : dict
            User configuration from YAML

        Returns:
        -------
        QEvalConfig
            Validated and merged configuration

        Raises:
        ------
        ConfigError
            If user configuration is invalid
        """
        try:
            # Convert base config to dict for merging
            base_dict = base_config.model_dump()

            # Recursively merge dictionaries
            merged_dict = self._deep_merge_dicts(base_dict, user_config)

            # Validate merged configuration with Pydantic
            return QEvalConfig.model_validate(merged_dict)

        except Exception as e:
            raise ConfigError(f"Configuration validation failed: {e}") from e

    def _deep_merge_dicts(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        """Recursively merge two dictionaries."""
        merged = base.copy()

        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value

        return merged

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key path.

        Parameters
        ----------
        key : str
            Dot-separated key path (e.g., 'evaluation.tier')
        default : Any, optional
            Default value if key not found

        Returns:
        -------
        Any
            Configuration value
        """
        keys = key.split(".")
        value = self.config.model_dump()

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def validate(self) -> None:
        """Validate the current configuration.

        This method is automatically called during initialization,
        but can be used to re-validate after manual modifications.

        Raises:
        ------
        ConfigError
            If configuration is invalid
        """
        try:
            # Pydantic validation happens automatically during model creation
            # This method is kept for API compatibility
            self.config.model_validate(self.config.model_dump())
        except Exception as e:
            raise ConfigError(f"Configuration validation failed: {e}") from e

    def save_to_yaml(self, config_path: str | Path) -> None:
        """Save current configuration to YAML file.

        Parameters
        ----------
        config_path : str or Path
            Path where to save the configuration
        """
        config_path = Path(config_path)

        try:
            with open(config_path, "w") as f:
                # Convert Pydantic model to dict and save as YAML
                config_dict = self.config.model_dump(exclude_none=True)
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        except OSError as e:
            raise ConfigError(f"Cannot write to {config_path}: {e}") from e

    def create_evaluator(self) -> Any:
        """Create Evaluator instance from configuration.

        Returns:
        -------
        ml4t-diagnostic.Evaluator
            Configured evaluator instance
        """
        from ml4t.diagnostic.evaluation.framework import Evaluator
        from ml4t.diagnostic.splitters import CombinatorialCV, WalkForwardCV

        # Create splitter
        splitter_type = self.config.splitter.type
        splitter_params = self.config.splitter.params.copy()

        # Add data-specific parameters
        if "label_horizon" not in splitter_params:
            splitter_params["label_horizon"] = self.config.data.label_horizon
        if "embargo_pct" not in splitter_params:
            splitter_params["embargo_pct"] = self.config.data.embargo_pct

        if splitter_type == "WalkForwardCV":
            splitter = WalkForwardCV(**splitter_params)
        else:  # CombinatorialCV
            splitter = CombinatorialCV(**splitter_params)

        # Get tier-specific configuration
        tier = self.config.evaluation.tier
        tier_key = cast(Literal["tier_1", "tier_2", "tier_3"], f"tier_{tier}")
        statistical_tests = self.config.statistical_tests[tier_key]

        # Create evaluator
        evaluator = Evaluator(
            splitter=splitter,
            metrics=list(self.config.metrics) if self.config.metrics else None,
            statistical_tests=statistical_tests,
            tier=tier,
            confidence_level=self.config.evaluation.confidence_level,
            bootstrap_samples=self.config.evaluation.bootstrap_samples,
            random_state=self.config.evaluation.random_state,
            n_jobs=self.config.evaluation.n_jobs,
        )

        return evaluator

    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"EvaluationConfigManager(tier={self.config.evaluation.tier}, metrics={self.config.metrics})"


# Explicit legacy alias to distinguish from `ml4t.diagnostic.config` managers.
LegacyEvaluationConfigManager = EvaluationConfigManager

# Backward compatibility alias
EvaluationConfig = EvaluationConfigManager


def load_config(
    config_path: str | Path | None = None,
) -> EvaluationConfigManager:
    """Load configuration from file or environment.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to configuration file. If None, checks QEVAL_CONFIG
        environment variable, then looks for ml4t-diagnostic.yaml in current
        directory.

    Returns:
    -------
    EvaluationConfigManager
        Loaded configuration
    """
    if config_path is None:
        # Check environment variable
        config_path = os.environ.get("QEVAL_CONFIG")

        if config_path is None:
            # Check current directory
            default_path = Path("mlquant.evaluation.yaml")
            if default_path.exists():
                config_path = default_path

    return EvaluationConfigManager(config_path)


# Example configuration template
EXAMPLE_CONFIG = """# ml4t-diagnostic Configuration File
# =======================

evaluation:
  tier: 2                    # Validation tier (1, 2, or 3)
  confidence_level: 0.05     # Significance level for tests
  bootstrap_samples: 1000    # Number of bootstrap samples
  random_state: 42          # Random seed for reproducibility
  n_jobs: 1                 # Number of parallel jobs

splitter:
  type: WalkForwardCV  # or CombinatorialCV
  params:
    n_splits: 5
    test_size: 0.2
    gap: 0
    expanding: true
    calendar: NYSE      # Default. Options: NYSE, CME_Equity, LSE, NASDAQ, etc.

metrics:
  - ic
  - sharpe
  - hit_rate
  - max_drawdown

statistical_tests:
  tier_1:
    - dsr
    - fdr
  tier_2:
    - hac_ic
  tier_3: []

data:
  label_horizon: 20          # Forward-looking period for labels
  embargo_pct: 0.01         # Embargo as percentage of data
  min_samples_per_fold: 100  # Minimum samples per CV fold

visualization:
  theme: default            # Visualization theme
  export_format: html       # Output format (html, png, svg)
  include_dashboard: true   # Generate full dashboard

logging:
  level: INFO
  use_wandb: false
  wandb_project: null
  wandb_entity: null
"""


def create_example_config(output_path: str | Path = "mlquant.evaluation.yaml") -> None:
    """Create an example configuration file.

    Parameters
    ----------
    output_path : str or Path
        Path for example configuration file
    """
    with open(output_path, "w") as f:
        f.write(EXAMPLE_CONFIG)
    print(f"Example configuration created at: {output_path}")


__all__ = [
    "ConfigError",
    "SplitterConfig",
    "DataConfig",
    "VisualizationConfig",
    "LoggingConfig",
    "EvaluatorConfig",
    "QEvalConfig",
    "LegacyEvaluationConfig",
    "EvaluationConfigManager",
    "LegacyEvaluationConfigManager",
    "EvaluationConfig",
    "load_config",
    "create_example_config",
]
