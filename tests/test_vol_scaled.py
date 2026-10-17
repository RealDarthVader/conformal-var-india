"""Unit tests for volatility-scaled conformal calibration."""

import numpy as np
import pandas as pd

from conformal_var.conformal.vol_scaled import (
    VolScaledConformalVaR,
    compute_vol_scaled_residual,
)


def test_vol_scaled_residual_formula() -> None:
    """Verify s_t^{scaled} = (q_hat - r_t) / sigma_t."""
    q_hat = -0.02
    r_t = -0.04
    sigma_1 = 0.01
    sigma_2 = 0.02

    # Violation (r_t < q_hat) yields positive residual
    s1 = compute_vol_scaled_residual(q_hat, r_t, sigma_1)
    s2 = compute_vol_scaled_residual(q_hat, r_t, sigma_2)

    assert s1 > 0.0
    assert np.isclose(s1, (-0.02 - (-0.04)) / 0.01)  # 0.02 / 0.01 = 2.0
    assert np.isclose(s2, (-0.02 - (-0.04)) / 0.02)  # 0.02 / 0.02 = 1.0
    assert np.isclose(s1, 2.0 * s2)


def test_vol_scaled_conformal_var_prediction_formula() -> None:
    """Verify VaR_t = q_hat - sigma_t * Q_{1-alpha}(s^{scaled})."""
    calibrator = VolScaledConformalVaR(alpha=0.05, window=50, adaptive=False)

    # Pre-populate residuals with known values: 50 values from 0.0 to 1.0
    known_residuals = np.linspace(0.0, 1.0, 50)
    base_preds = np.zeros(50)
    sigmas = np.ones(50)
    returns = -known_residuals  # so s = (0 - (-known)) / 1 = known

    calibrator.calibrate(base_preds, returns, sigmas)

    thresh = calibrator.current_threshold()
    sigma_t = 0.025
    q_base = -0.03
    expected_var = q_base - sigma_t * thresh

    pred_var = calibrator.predict_var(q_base, sigma_t)
    assert np.isclose(pred_var, expected_var)


def test_vol_scaled_aci_adaptation_dynamics() -> None:
    """Verify online gradient step: err_t=1 decreases alpha_t, err_t=0 increases alpha_t."""
    calibrator = VolScaledConformalVaR(alpha=0.05, gamma=0.01, window=50, adaptive=True)
    init_alpha = calibrator.alpha_t

    # Trigger a violation: return is far below predicted VaR
    calibrator.predict_var(-0.02, sigma=0.015)
    _, next_alpha, err = calibrator.update(
        base_prediction=-0.02,
        realized_return=-0.10,
        sigma=0.015,
    )
    assert err == 1
    assert next_alpha < init_alpha

    # Trigger safe day: return is positive
    cur_alpha = calibrator.alpha_t
    calibrator.predict_var(-0.02, sigma=0.015)
    _, next_alpha_safe, err_safe = calibrator.update(
        base_prediction=-0.02,
        realized_return=0.02,
        sigma=0.015,
    )
    assert err_safe == 0
    assert next_alpha_safe > cur_alpha


def test_vol_scaled_predict_sequence() -> None:
    """Verify sequential execution produces valid DataFrame with all columns."""
    n = 100
    base_preds = pd.Series(np.full(n, -0.025))
    returns = pd.Series(np.random.normal(0, 0.015, n))
    sigmas = pd.Series(np.full(n, 0.015))

    calibrator = VolScaledConformalVaR(alpha=0.05, window=30)
    df = calibrator.predict_sequence(base_preds, returns, sigmas)

    assert len(df) == n
    assert "conformal_var" in df.columns
    assert "scaled_residual" in df.columns
    assert "threshold" in df.columns
    assert "violation" in df.columns
    assert "sigma" in df.columns
