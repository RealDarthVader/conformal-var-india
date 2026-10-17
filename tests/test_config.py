"""Unit and validation tests for conformal-var configuration schema."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from conformal_var.config import (
    AppConfig,
    MarketRegime,
    WindowConfig,
    load_config,
)


def test_load_default_config() -> None:
    """Verify loading from the default config.yaml matches all required specifications."""
    config = load_config()

    # Target quantiles and derived confidence levels
    assert config.target_quantiles == [0.05, 0.01]
    assert config.confidence_levels == [0.95, 0.99]

    # Windows
    assert config.windows.training_window == 756
    assert config.windows.calibration_window == 252

    # Required Indian Market Regimes
    regimes = config.market_regimes
    assert "calm_bull" in regimes
    assert "ilfs_liquidity_crisis" in regimes
    assert "covid_crash" in regimes
    assert "rate_hike_tightening" in regimes

    assert regimes["calm_bull"].start_date == date(2017, 1, 1)
    assert regimes["calm_bull"].end_date == date(2018, 1, 31)

    assert regimes["ilfs_liquidity_crisis"].start_date == date(2018, 9, 1)
    assert regimes["ilfs_liquidity_crisis"].end_date == date(2019, 3, 31)

    assert regimes["covid_crash"].start_date == date(2020, 2, 1)
    assert regimes["covid_crash"].end_date == date(2020, 6, 30)

    assert regimes["rate_hike_tightening"].start_date == date(2022, 1, 1)
    assert regimes["rate_hike_tightening"].end_date == date(2022, 12, 31)

    # get_regime helper
    calm = config.get_regime("calm_bull")
    assert calm.start_date == date(2017, 1, 1)


def test_missing_regime_raises_key_error() -> None:
    """Verify accessing a non-existent regime raises KeyError with helpful message."""
    config = load_config()
    with pytest.raises(KeyError, match="Regime 'unknown_shock' not found"):
        config.get_regime("unknown_shock")


@pytest.mark.parametrize("invalid_q", [
    [-0.05],
    [0.0],
    [1.0],
    [1.05],
    [],
])
def test_invalid_quantiles_validation(invalid_q: list[float]) -> None:
    """Ensure quantiles outside (0, 1) or empty lists are rejected."""
    with pytest.raises(ValidationError):
        AppConfig(target_quantiles=invalid_q)


def test_invalid_window_sizes() -> None:
    """Ensure calibration window cannot exceed or equal training window."""
    msg = "calibration_window .* must be less than training_window"
    with pytest.raises(ValidationError, match=msg):
        WindowConfig(training_window=252, calibration_window=252)

    with pytest.raises(ValidationError):
        WindowConfig(training_window=100, calibration_window=200)

    with pytest.raises(ValidationError):
        WindowConfig(training_window=0, calibration_window=50)


def test_invalid_market_regime_dates() -> None:
    """Ensure start_date > end_date triggers validation error."""
    with pytest.raises(ValidationError, match="start_date .* must be <= end_date"):
        MarketRegime(
            start_date=date(2020, 6, 30),
            end_date=date(2020, 2, 1),
        )


def test_extra_fields_forbidden() -> None:
    """Ensure strict schema forbids undeclared extra fields."""
    with pytest.raises(ValidationError):
        AppConfig.model_validate({
            "target_quantiles": [0.05, 0.01],
            "unexpected_field": "disallowed",
        })

    with pytest.raises(ValidationError):
        WindowConfig.model_validate({
            "training_window": 756,
            "calibration_window": 252,
            "extra_window": 100,
        })


def test_frozen_immutability() -> None:
    """Ensure config objects are frozen and immutable at runtime."""
    config = load_config()
    with pytest.raises(ValidationError):
        config.windows.training_window = 500  # type: ignore[misc]


def test_nonexistent_config_file(tmp_path: Path) -> None:
    """Ensure missing file raises FileNotFoundError."""
    dummy_file = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(dummy_file)


def test_invalid_yaml_structure(tmp_path: Path) -> None:
    """Ensure YAML with invalid root structure raises TypeError."""
    bad_yaml = tmp_path / "bad_config.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Expected YAML file to contain a mapping/dict"):
        load_config(bad_yaml)
