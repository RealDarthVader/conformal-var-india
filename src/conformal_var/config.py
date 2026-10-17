"""Strict configuration schema and loader for conformal-var-india.

Provides Pydantic models with strict validation for target risk quantiles,
rolling estimation & calibration windows, and Indian market crisis regimes.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MarketRegime(BaseModel):
    """Represents a specific market crisis or regime period with exact dates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_date_ordering(self) -> MarketRegime:
        """Ensure start_date is before or equal to end_date."""
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be <= end_date ({self.end_date})"
            )
        return self


class WindowConfig(BaseModel):
    """Rolling estimation and calibration window sizes in trading days."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    training_window: int = Field(
        default=756,
        description="Rolling training window in trading days (~3 years).",
        gt=0,
    )
    calibration_window: int = Field(
        default=252,
        description="Conformal calibration window in trading days (~1 year).",
        gt=0,
    )

    @model_validator(mode="after")
    def validate_window_sizes(self) -> WindowConfig:
        """Validate window sizing hierarchy."""
        if self.calibration_window >= self.training_window:
            raise ValueError(
                f"calibration_window ({self.calibration_window}) must be less than "
                f"training_window ({self.training_window})"
            )
        return self


class DataConfig(BaseModel):
    """Configuration for data ingestion and artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_tickers: list[str] = Field(
        default_factory=lambda: ["^NSEI", "^NSEBANK"],
        description="Default NSE index ticker symbols.",
    )
    storage_format: str = Field(
        default="parquet",
        description="Storage format for processed tabular time series.",
    )
    artifacts_dir: Path = Field(
        default=Path("artifacts"),
        description="Base directory for artifacts.",
    )


class AppConfig(BaseModel):
    """Master application configuration with strict schema validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_quantiles: list[float] = Field(
        default_factory=lambda: [0.05, 0.01],
        description="Target quantiles alpha in {0.05, 0.01} representing 95% and 99% VaR.",
    )
    windows: WindowConfig = Field(default_factory=WindowConfig)
    market_regimes: dict[str, MarketRegime] = Field(default_factory=dict)
    data: DataConfig = Field(default_factory=DataConfig)

    @field_validator("target_quantiles")
    @classmethod
    def validate_target_quantiles(cls, quantiles: list[float]) -> list[float]:
        """Validate target quantiles are strictly in (0, 1) and non-empty."""
        if not quantiles:
            raise ValueError("target_quantiles list cannot be empty")
        for q in quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError(f"Each quantile must be strictly in (0, 1), got {q}")
        return sorted(quantiles, reverse=True)

    @property
    def confidence_levels(self) -> list[float]:
        """Return corresponding confidence levels 1 - alpha."""
        return [round(1.0 - q, 4) for q in self.target_quantiles]

    def get_regime(self, regime_name: str) -> MarketRegime:
        """Fetch regime by name with descriptive key error if missing."""
        if regime_name not in self.market_regimes:
            available = ", ".join(self.market_regimes.keys())
            raise KeyError(f"Regime '{regime_name}' not found. Available regimes: {available}")
        return self.market_regimes[regime_name]


def find_default_config_path() -> Path:
    """Find config.yaml in current directory or project root."""
    candidate_paths = [
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent.parent.parent / "config.yaml",
    ]
    for p in candidate_paths:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find config.yaml in current directory or project root."
    )


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str | Path | None
        Path to config.yaml. If None, auto-discovers config.yaml.

    Returns
    -------
    AppConfig
        Validated, immutable application configuration.
    """
    resolved_path: Path
    if config_path is None:
        resolved_path = find_default_config_path()
    else:
        resolved_path = Path(config_path)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as f:
        raw_data: Any = yaml.safe_load(f)

    if raw_data is None:
        raw_data = {}

    if not isinstance(raw_data, dict):
        raise TypeError(
            f"Expected YAML file to contain a mapping/dict, got {type(raw_data).__name__}"
        )

    return AppConfig.model_validate(raw_data)
