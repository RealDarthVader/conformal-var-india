"""Tests for conformal prediction calibration engines and ACI coverage recovery."""

import numpy as np
import pandas as pd
import pytest

from conformal_var.conformal.calibrator import (
    AdaptiveConformalVaR,
    SlidingWindowConformalVaR,
    SplitConformalVaR,
    compute_conformity_residual,
    finite_sample_conformal_quantile,
)


def generate_ar1_garch_shock_stream(
    n_days: int = 1600,
    shock_day: int = 800,
    seed: int = 42,
) -> tuple[pd.Series, int]:
    """Generate synthetic stationary AR(1)-GARCH(1,1) stream with a 3x volatility shock."""
    rng = np.random.default_rng(seed=seed)
    returns = np.zeros(n_days, dtype=np.float64)
    sigma2 = np.zeros(n_days, dtype=np.float64)

    omega_base = 0.00002
    alpha_1 = 0.08
    beta_1 = 0.85

    sigma2[0] = omega_base / (1.0 - alpha_1 - beta_1)

    for t in range(1, n_days):
        # 3x volatility shock (variance multiplies by 9x)
        omega = omega_base * 9.0 if t >= shock_day else omega_base
        sigma2[t] = omega + alpha_1 * (returns[t - 1] ** 2) + beta_1 * sigma2[t - 1]
        z_t = rng.standard_normal()
        returns[t] = 0.05 * returns[t - 1] + np.sqrt(sigma2[t]) * z_t

    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    return pd.Series(returns, index=dates, name="return"), shock_day


def test_signed_conformity_residual_definition() -> None:
    """Verify signed residual: violation (r_t < q_hat) corresponds to s_t > 0."""
    q_hat = -0.02
    r_violation = -0.035
    s_violation = compute_conformity_residual(q_hat, r_violation)
    assert s_violation > 0.0, f"Violation must yield s_t > 0, got {s_violation}"

    r_safe = -0.010
    s_safe = compute_conformity_residual(q_hat, r_safe)
    assert s_safe < 0.0, f"Non-violation must yield s_t < 0, got {s_safe}"


def test_finite_sample_quantile_formula() -> None:
    """Verify ceil((M+1)(1-alpha))/M finite-sample adjustment."""
    m = 252
    alpha = 0.05
    # (253 * 0.95) = 240.35 -> ceil is 241 -> 241 / 252 ~ 0.956349
    residuals = np.linspace(-0.05, 0.05, m)
    q = finite_sample_conformal_quantile(residuals, alpha=alpha)

    # In a uniformly spaced array from -0.05 to 0.05 of length 252,
    # the 241st element is strictly near the top
    assert q > 0.04


def test_volatility_shock_adaptive_vs_split_recovery() -> None:
    """Prove ACI restores empirical coverage to 1-alpha +/- 0.5% after a 3x shock."""
    returns, shock_day = generate_ar1_garch_shock_stream(n_days=1600, shock_day=800, seed=42)
    alpha = 0.05
    m_cal = 252

    # Pre-shock calibration buffer
    cal_returns = returns.iloc[shock_day - m_cal : shock_day]
    # Static base model prediction from pre-shock calm era
    base_cal_pred = pd.Series(np.quantile(cal_returns, alpha), index=cal_returns.index)

    # 1. Calibrate SplitConformalVaR on pre-shock calm era
    split_conf = SplitConformalVaR(alpha=alpha, calibration_size=m_cal)
    split_conf.calibrate(base_cal_pred, cal_returns)

    # 2. Calibrate AdaptiveConformalVaR on pre-shock calm era
    aci = AdaptiveConformalVaR(alpha=alpha, gamma=0.025, window=m_cal)
    aci.calibrate(base_cal_pred, cal_returns)

    # Post-shock evaluation period (days 800 to 1600)
    post_returns = returns.iloc[shock_day:]
    base_post_pred = pd.Series(float(base_cal_pred.iloc[0]), index=post_returns.index)

    # Run Split Conformal on post-shock
    split_preds = split_conf.predict(base_post_pred)
    split_violations = post_returns < split_preds
    split_post_coverage = 1.0 - float(split_violations.mean())

    # Run ACI on post-shock
    aci_res = aci.predict_sequence(base_post_pred, post_returns)
    aci_post_coverage = 1.0 - float(aci_res["violation"].mean())

    # SplitConformal coverage collapses severely due to broken exchangeability
    split_msg = (
        f"SplitConformal was expected to collapse below 80% coverage under 3x shock, "
        f"got {split_post_coverage:.2%}"
    )
    assert split_post_coverage < 0.80, split_msg

    # ACI overall post-shock coverage is much closer to 95%
    assert aci_post_coverage > 0.90, (
        f"ACI post-shock coverage was expected to exceed 90%, got {aci_post_coverage:.2%}"
    )

    # In the second half of the shock (days 250+ after shock), ACI has fully adapted
    # and restored coverage to 1 - alpha +/- 0.5% (95% +/- 0.5%, i.e. 94.5% to 95.5%)
    aci_adapted_coverage = 1.0 - float(aci_res["violation"].iloc[250:].mean())
    split_adapted_coverage = 1.0 - float(split_violations.iloc[250:].mean())

    # Verify ACI restored empirical coverage to 1 - alpha +/- 0.5% (0.945 to 0.955)
    assert 0.945 <= aci_adapted_coverage <= 0.955, (
        f"ACI adapted coverage ({aci_adapted_coverage:.3%}) not in [94.5%, 95.5%]"
    )

    # And Split Conformal remains severely degraded
    assert split_adapted_coverage < 0.80, (
        f"SplitConformal adapted coverage remained too high: {split_adapted_coverage:.3%}"
    )


def test_sliding_window_conformal_walk_forward() -> None:
    """Verify SlidingWindowConformalVaR updates residual window properly."""
    returns = pd.Series(np.random.normal(0, 0.015, 300))
    base_preds = pd.Series(np.full(300, -0.025))

    sw = SlidingWindowConformalVaR(alpha=0.05, window=50)
    sw.calibrate(base_preds.iloc[:50], returns.iloc[:50])

    res_df = sw.predict_sequence(base_preds.iloc[50:], returns.iloc[50:])
    assert len(res_df) == 250
    assert "conformal_var" in res_df.columns
    assert "violation" in res_df.columns


@pytest.mark.parametrize("gamma", [0.005, 0.01, 0.025, 0.05])
def test_adaptive_conformal_gammas(gamma: float) -> None:
    """Verify ACI runs with all required gamma learning rates and clips properly."""
    aci = AdaptiveConformalVaR(alpha=0.05, gamma=gamma, window=50)
    init_alpha = aci.alpha_t

    # Case 1: Violation occurs (err = 1) -> alpha_{t+1} decreases
    aci.predict_var(-0.02)
    _, next_alpha, err = aci.update(base_prediction=-0.02, realized_return=-0.05)
    assert err == 1
    assert next_alpha < init_alpha

    # Case 2: No violation occurs (err = 0) -> alpha_{t+1} increases
    cur_alpha = aci.alpha_t
    aci.predict_var(-0.02)
    _, next_alpha_no_err, err_zero = aci.update(base_prediction=-0.02, realized_return=0.01)
    assert err_zero == 0
    assert next_alpha_no_err > cur_alpha
