"""Tests for institutional 1-step-ahead Value-at-Risk (VaR) baseline forecasters."""

import numpy as np
import pandas as pd
import pytest

from conformal_var.models.baselines import (
    BaseVaRForecaster,
    GARCHNormalVaR,
    GARCHStudentTVaR,
    HistoricalSimulationVaR,
    RiskMetricsEWMAVaR,
)


@pytest.fixture
def synthetic_returns() -> pd.Series:
    """Fixture providing 1000 days of realistic daily log returns."""
    rng = np.random.default_rng(seed=42)
    # Generate realistic returns with mild volatility clustering
    innovations = rng.standard_t(df=5, size=1000) * 0.012
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    return pd.Series(innovations, index=dates, name="return")


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [
        (HistoricalSimulationVaR, {"window": 252}),
        (HistoricalSimulationVaR, {"window": 500}),
        (RiskMetricsEWMAVaR, {"decay_factor": 0.94}),
        (GARCHNormalVaR, {"training_window": 756, "refit_every": 21}),
        (GARCHStudentTVaR, {"training_window": 756, "refit_every": 21, "dist": "t"}),
        (GARCHStudentTVaR, {"training_window": 756, "refit_every": 21, "dist": "skewt"}),
    ],
)
def test_strict_lookahead_leakage(
    synthetic_returns: pd.Series,
    model_cls: type[BaseVaRForecaster],
    kwargs: dict[str, object],
) -> None:
    """Verify that modifying return r_t NEVER alters VaR_t(alpha) (strict causality)."""
    t_idx = 850
    returns_original = synthetic_returns.copy()

    # 1. Fit model on history up to t-1
    model1 = model_cls(**kwargs)
    model1.fit(returns_original.iloc[:t_idx])
    var_95_orig = model1.predict_var(0.05)
    var_99_orig = model1.predict_var(0.01)

    # 2. Mutate return at day t to extreme positive shock (+25%)
    returns_shocked_up = returns_original.copy()
    returns_shocked_up.iloc[t_idx] = 0.25

    model2 = model_cls(**kwargs)
    model2.fit(returns_shocked_up.iloc[:t_idx])
    var_95_shock_up = model2.predict_var(0.05)
    var_99_shock_up = model2.predict_var(0.01)

    assert np.isclose(var_95_orig, var_95_shock_up, atol=1e-12), (
        f"{model_cls.__name__} leaked future data: positive shock altered VaR_t"
    )
    assert np.isclose(var_99_orig, var_99_shock_up, atol=1e-12)

    # 3. Mutate return at day t to extreme crash (-50%)
    returns_shocked_down = returns_original.copy()
    returns_shocked_down.iloc[t_idx] = -0.50

    model3 = model_cls(**kwargs)
    model3.fit(returns_shocked_down.iloc[:t_idx])
    var_95_shock_down = model3.predict_var(0.05)
    var_99_shock_down = model3.predict_var(0.01)

    assert np.isclose(var_95_orig, var_95_shock_down, atol=1e-12), (
        f"{model_cls.__name__} leaked future data: negative shock altered VaR_t"
    )
    assert np.isclose(var_99_orig, var_99_shock_down, atol=1e-12)


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [
        (HistoricalSimulationVaR, {"window": 252}),
        (HistoricalSimulationVaR, {"window": 500}),
        (RiskMetricsEWMAVaR, {"decay_factor": 0.94}),
        (GARCHNormalVaR, {"training_window": 756}),
        (GARCHStudentTVaR, {"training_window": 756, "dist": "t"}),
        (GARCHStudentTVaR, {"training_window": 756, "dist": "skewt"}),
    ],
)
def test_99_var_strictly_more_negative_than_95_var(
    synthetic_returns: pd.Series,
    model_cls: type[BaseVaRForecaster],
    kwargs: dict[str, object],
) -> None:
    """Verify that 99% VaR is strictly more negative than 95% VaR (monotonicity)."""
    model = model_cls(**kwargs)
    model.fit(synthetic_returns.iloc[:800])

    var_95 = model.predict_var(0.05)
    var_99 = model.predict_var(0.01)

    # Both must represent downside loss (negative return quantiles)
    assert var_95 < 0.0, f"{model_cls.__name__}: 95% VaR should be negative, got {var_95}"
    assert var_99 < 0.0, f"{model_cls.__name__}: 99% VaR should be negative, got {var_99}"

    # 99% VaR must be strictly more severe (more negative) than 95% VaR
    assert var_99 < var_95, (
        f"{model_cls.__name__}: expected VaR(0.01) < VaR(0.05), got {var_99} >= {var_95}"
    )


def test_predict_rolling_leakage(synthetic_returns: pd.Series) -> None:
    """Verify that predict_rolling outputs at day t depend strictly on past returns."""
    model = RiskMetricsEWMAVaR(decay_factor=0.94)
    res_orig = model.predict_rolling(
        synthetic_returns.iloc[:300],
        alphas=(0.05, 0.01),
        min_history=252,
    )

    # Mutate the last observation in the series
    mutated = synthetic_returns.iloc[:300].copy()
    mutated.iloc[-1] = -0.99
    res_mut = model.predict_rolling(mutated, alphas=(0.05, 0.01), min_history=252)

    # Forecast for the last day must be completely unchanged because it used data up to -2
    last_dt = res_orig.index[-1]
    assert np.isclose(res_orig.loc[last_dt, "VaR_95"], res_mut.loc[last_dt, "VaR_95"], atol=1e-12)


def test_historical_simulation_empirical_quantile(synthetic_returns: pd.Series) -> None:
    """Verify HistoricalSimulationVaR exactly matches empirical window quantile."""
    window = 252
    model = HistoricalSimulationVaR(window=window)
    hist_slice = synthetic_returns.iloc[:300]
    model.fit(hist_slice)

    expected_05 = float(np.quantile(hist_slice.iloc[-window:], 0.05, method="linear"))
    expected_01 = float(np.quantile(hist_slice.iloc[-window:], 0.01, method="linear"))

    assert np.isclose(model.predict_var(0.05), expected_05)
    assert np.isclose(model.predict_var(0.01), expected_01)


def test_riskmetrics_recurrence_verification() -> None:
    """Verify RiskMetrics EWMA follows:

    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2.
    """
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02], dtype=np.float64)
    lam = 0.94
    model = RiskMetricsEWMAVaR(decay_factor=lam)
    model.fit(returns)

    # Manual recurrence:
    var_0 = returns[0] ** 2
    var_1 = lam * var_0 + (1 - lam) * (returns[1] ** 2)
    var_2 = lam * var_1 + (1 - lam) * (returns[2] ** 2)
    var_3 = lam * var_2 + (1 - lam) * (returns[3] ** 2)
    var_4 = lam * var_3 + (1 - lam) * (returns[4] ** 2)
    expected_sigma = np.sqrt(var_4)

    from scipy.stats import norm
    expected_var_95 = norm.ppf(0.05) * expected_sigma

    assert np.isclose(model.predict_var(0.05), expected_var_95, atol=1e-10)


def test_invalid_arguments_and_unfitted_state(synthetic_returns: pd.Series) -> None:
    """Verify proper validation errors on invalid inputs and unfitted state."""
    model = HistoricalSimulationVaR(window=252)

    # Calling predict before fit
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.predict_var(0.05)

    # Insufficient observations
    with pytest.raises(ValueError, match="requires at least 252 returns"):
        model.fit(synthetic_returns.iloc[:50])

    model.fit(synthetic_returns.iloc[:300])

    # Invalid alpha values
    with pytest.raises(ValueError, match="must be in"):
        model.predict_var(0.0)

    with pytest.raises(ValueError, match="must be in"):
        model.predict_var(1.0)

    with pytest.raises(ValueError, match="must be in"):
        model.predict_var(-0.05)
