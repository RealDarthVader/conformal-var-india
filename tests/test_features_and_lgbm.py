"""Tests for causal feature engineering and LightGBM Quantile VaR engine."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_pinball_loss

from conformal_var.features.engineer import FEATURE_NAMES, FeatureEngineer
from conformal_var.models.lgbm_quantile import LightGBMQuantileVaR


@pytest.fixture
def synthetic_market_data() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Generate realistic OHLCV, returns, and market indicators for testing."""
    rng = np.random.default_rng(seed=123)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    # Asset prices & returns
    log_returns = rng.normal(0.0005, 0.015, n)
    returns = pd.Series(log_returns, index=dates, name="asset_returns")
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    high = pd.Series(prices * (1.0 + np.abs(rng.normal(0, 0.008, n))), index=dates, name="High")
    low = pd.Series(prices * (1.0 - np.abs(rng.normal(0, 0.008, n))), index=dates, name="Low")

    # Market context
    mkt_returns = pd.Series(rng.normal(0.0003, 0.012, n), index=dates, name="^NSEI")
    vix = pd.Series(15.0 + np.abs(rng.normal(0, 3.0, n)), index=dates, name="^INDIAVIX")

    return returns, high, low, mkt_returns, vix


def test_causal_shifting_leakage_immunity(
    synthetic_market_data: tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series],
) -> None:
    """Verify X.loc[t] strictly depends only on information at or before t-1."""
    returns, high, low, mkt_ret, vix = synthetic_market_data
    engineer = FeatureEngineer()

    # 1. Compute original feature matrix
    X_orig = engineer.compute_features(
        asset_returns=returns,
        asset_high=high,
        asset_low=low,
        market_returns=mkt_ret,
        vix_prices=vix,
    )

    t_idx = 150
    t_date = returns.index[t_idx]
    orig_row_t = X_orig.loc[t_date].copy()

    # 2. Mutate returns, high, low, and market data at day t and subsequent days
    returns_mut = returns.copy()
    high_mut = high.copy()
    low_mut = low.copy()
    mkt_ret_mut = mkt_ret.copy()
    vix_mut = vix.copy()

    returns_mut.iloc[t_idx:] += 0.30
    high_mut.iloc[t_idx:] *= 2.0
    low_mut.iloc[t_idx:] *= 0.5
    mkt_ret_mut.iloc[t_idx:] -= 0.20
    vix_mut.iloc[t_idx:] += 40.0

    X_mutated = engineer.compute_features(
        asset_returns=returns_mut,
        asset_high=high_mut,
        asset_low=low_mut,
        market_returns=mkt_ret_mut,
        vix_prices=vix_mut,
    )
    mut_row_t = X_mutated.loc[t_date]

    # Every single feature at index t must be completely identical
    for col in FEATURE_NAMES:
        val_orig = orig_row_t[col]
        val_mut = mut_row_t[col]
        if np.isnan(val_orig):
            assert np.isnan(val_mut), f"NaN mismatch at {col}"
        else:
            assert np.isclose(val_orig, val_mut, atol=1e-10), (
                f"Lookahead leakage detected in '{col}': modifying data at t altered X.loc[t]"
            )

    # 3. Assert all rows up to t are identical
    pd.testing.assert_frame_equal(X_orig.iloc[: t_idx + 1], X_mutated.iloc[: t_idx + 1])


def test_pinball_loss_decreases_on_heteroscedastic_data() -> None:
    """Verify LightGBMQuantileVaR achieves lower pinball loss on heteroscedastic data."""
    rng = np.random.default_rng(seed=42)
    n = 1200
    p = 5

    # Features: volatility regime is strongly driven by x_0
    X_mat = rng.normal(0, 1, size=(n, p))
    regime = (X_mat[:, 0] > 0).astype(float)
    sigma = 0.01 + 0.04 * regime

    # Heteroscedastic returns
    y_vec = rng.normal(0, 1, size=n) * sigma

    X_df = pd.DataFrame(X_mat, columns=[f"feat_{i}" for i in range(p)])
    y_ser = pd.Series(y_vec, name="target")

    # Train/test split (800 train, 400 test)
    X_train, y_train = X_df.iloc[:800], y_ser.iloc[:800]
    X_test, y_test = X_df.iloc[800:], y_ser.iloc[800:]

    lgb_var = LightGBMQuantileVaR(
        alphas=(0.05, 0.01),
        max_depth=4,
        num_leaves=15,
        min_child_samples=30,
        learning_rate=0.03,
        n_estimators=100,
        random_state=42,
    )
    lgb_var.fit(X_train, y_train)

    # Predict test quantiles
    preds_05 = [lgb_var.predict_var(X_test.iloc[[i]], alpha=0.05) for i in range(len(X_test))]
    preds_05_arr = np.array(preds_05)

    # Baseline: unconditional empirical 5% quantile from training set
    base_q05 = float(np.quantile(y_train, 0.05))
    base_preds_arr = np.full_like(preds_05_arr, base_q05)

    loss_lgb = mean_pinball_loss(y_test, preds_05_arr, alpha=0.05)
    loss_base = mean_pinball_loss(y_test, base_preds_arr, alpha=0.05)

    # LightGBM must significantly beat static baseline
    assert loss_lgb < loss_base, (
        f"LGBM pinball loss ({loss_lgb:.5f}) not strictly lower than baseline ({loss_base:.5f})"
    )
    # Check at least 15% reduction in pinball loss
    pct_improvement = (loss_base - loss_lgb) / loss_base
    msg = f"Expected >15% pinball loss improvement, got {pct_improvement:.2%}"
    assert pct_improvement > 0.15, msg


def test_lgbm_monotonicity_99_more_negative_than_95() -> None:
    """Verify that 99% VaR is strictly more negative than 95% VaR across test samples."""
    rng = np.random.default_rng(seed=999)
    n = 600
    X = pd.DataFrame(rng.normal(0, 1, (n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(rng.normal(0, 0.02, n))

    model = LightGBMQuantileVaR(
        alphas=(0.05, 0.01),
        max_depth=3,
        num_leaves=8,
        min_child_samples=20,
        learning_rate=0.05,
        n_estimators=80,
    )
    model.fit(X, y)

    # Check across multiple evaluation points
    for i in range(20):
        sample = X.iloc[[i]]
        v95 = model.predict_var(sample, alpha=0.05)
        v99 = model.predict_var(sample, alpha=0.01)

        assert v95 < 0.0, f"VaR(0.05) should be negative, got {v95}"
        assert v99 < 0.0, f"VaR(0.01) should be negative, got {v99}"
        assert v99 < v95, f"Expected VaR(0.01) < VaR(0.05), got {v99} >= {v95}"


def test_walk_forward_rolling_and_calibration_buffer() -> None:
    """Verify walk-forward rolling retraining and calibration buffer isolation."""
    rng = np.random.default_rng(seed=777)
    n = 350
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    X = pd.DataFrame(rng.normal(0, 1, (n, 3)), index=dates, columns=["f1", "f2", "f3"])
    y = pd.Series(rng.normal(0, 0.02, n), index=dates)

    training_window = 120
    calib_buffer = 40
    refit_every = 20

    model = LightGBMQuantileVaR(
        alphas=(0.05, 0.01),
        training_window=training_window,
        calibration_buffer=calib_buffer,
        refit_every=refit_every,
        min_child_samples=15,
        n_estimators=30,
    )

    roll_preds = model.predict_rolling(X, y)

    # Starts forecasting at index training_window
    expected_len = n - training_window
    assert len(roll_preds) == expected_len
    assert "VaR_95" in roll_preds.columns
    assert "VaR_99" in roll_preds.columns

    # Verify calibration buffer data is preserved and accessible
    cal_data = model.calibration_buffer_data
    assert cal_data is not None
    X_cal, y_cal = cal_data
    assert len(X_cal) == calib_buffer
    assert len(y_cal) == calib_buffer
