"""Volatility-scaled Conformal Prediction for Value-at-Risk.

Normalizes signed conformity scores by rolling realized or conditional volatility:
    s_t^{scaled} = (q_hat_alpha(X_t) - r_t) / sigma_t
    VaR_t^{scaled} = q_hat_alpha(X_t) - sigma_t * Q_{1 - alpha_t}(s^{scaled})

Stabilizes nonconformity distributions across heteroscedastic regimes and market panics.
Supports both sliding-window CQR and online Adaptive Conformal Inference (ACI).
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from conformal_var.conformal.calibrator import finite_sample_conformal_quantile


def compute_vol_scaled_residual(
    base_prediction: float | npt.NDArray[np.float64] | pd.Series,
    realized_return: float | npt.NDArray[np.float64] | pd.Series,
    sigma: float | npt.NDArray[np.float64] | pd.Series,
    eps: float = 1e-6,
) -> Any:
    """Compute volatility-normalized conformity residual.

    s_t^{scaled} = (q_hat(X_t) - r_t) / max(sigma_t, eps)
    """
    numerator = base_prediction - realized_return
    if isinstance(sigma, (np.ndarray, pd.Series)):
        denominator = np.maximum(sigma, eps)
    else:
        denominator = max(sigma, eps)
    return numerator / denominator


class VolScaledConformalVaR:
    """Volatility-scaled Conformal VaR calibrator with optional ACI adaptation.

    Normalizes residuals by local volatility sigma_t, preserving exchangeability
    under heteroscedastic regimes.
    """

    ALLOWED_GAMMAS: Sequence[float] = (0.005, 0.01, 0.025, 0.05)

    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 0.01,
        window: int = 252,
        adaptive: bool = True,
        alpha_min: float = 0.001,
        alpha_max: float = 0.999,
        eps: float = 1e-6,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        if window < 10:
            raise ValueError(f"window must be >= 10, got {window}")

        self.alpha = alpha
        self.gamma = gamma
        self.window = window
        self.adaptive = adaptive
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.eps = eps

        self.alpha_t: float = alpha
        self._residual_buffer: collections.deque[float] = collections.deque(maxlen=window)
        self._last_predicted_var: float | None = None

    def calibrate(
        self,
        base_predictions: pd.Series | npt.NDArray[np.float64],
        returns: pd.Series | npt.NDArray[np.float64],
        sigmas: pd.Series | npt.NDArray[np.float64],
    ) -> VolScaledConformalVaR:
        """Initialize scaled residual buffer using holdout calibration slice."""
        q_arr = np.asarray(base_predictions, dtype=np.float64).ravel()
        r_arr = np.asarray(returns, dtype=np.float64).ravel()
        s_arr = np.asarray(sigmas, dtype=np.float64).ravel()

        if not (len(q_arr) == len(r_arr) == len(s_arr)):
            raise ValueError(
                f"Length mismatch: q={len(q_arr)}, r={len(r_arr)}, sigma={len(s_arr)}"
            )

        scaled_residuals = compute_vol_scaled_residual(q_arr, r_arr, s_arr, eps=self.eps)
        self._residual_buffer.clear()
        for res in scaled_residuals[-self.window:]:
            self._residual_buffer.append(float(res))

        self.alpha_t = self.alpha
        return self

    def current_threshold(self) -> float:
        """Compute Q_{1-alpha_t}(s^{scaled}) over recent scaled residuals."""
        if not self._residual_buffer:
            return 0.0
        return finite_sample_conformal_quantile(list(self._residual_buffer), self.alpha_t)

    def predict_var(self, base_prediction: float, sigma: float) -> float:
        """Predict 1-step-ahead volatility-scaled conformal VaR for day t.

        VaR_t = q_hat(X_t) - sigma_t * Q_{1 - alpha_t}(s^{scaled})
        """
        thresh = self.current_threshold()
        pred = base_prediction - max(sigma, self.eps) * thresh
        self._last_predicted_var = pred
        return pred

    def update(
        self,
        base_prediction: float,
        realized_return: float,
        sigma: float,
        conformal_var: float | None = None,
    ) -> tuple[float, float, int]:
        """Incorporate realized return at market close of day t to update buffer and alpha_t."""
        c_var = conformal_var if conformal_var is not None else self._last_predicted_var
        if c_var is None:
            c_var = self.predict_var(base_prediction, sigma)

        err_t = 1 if realized_return < c_var else 0
        s_scaled = float(
            compute_vol_scaled_residual(base_prediction, realized_return, sigma, eps=self.eps)
        )
        self._residual_buffer.append(s_scaled)

        if self.adaptive:
            new_alpha = self.alpha_t + self.gamma * (self.alpha - float(err_t))
            self.alpha_t = float(np.clip(new_alpha, self.alpha_min, self.alpha_max))

        return s_scaled, self.alpha_t, err_t

    def predict_sequence(
        self,
        base_predictions: pd.Series,
        returns: pd.Series,
        sigmas: pd.Series,
    ) -> pd.DataFrame:
        """Run sequential online volatility-scaled conformal calibration across time."""
        if not (len(base_predictions) == len(returns) == len(sigmas)):
            raise ValueError("Length mismatch between predictions, returns, and sigmas")

        conformal_vars: list[float] = []
        alpha_ts: list[float] = []
        thresholds: list[float] = []
        violations: list[int] = []
        scaled_residuals: list[float] = []

        for q_t, r_t, sig_t in zip(base_predictions, returns, sigmas, strict=True):
            alpha_ts.append(self.alpha_t)
            thresh = self.current_threshold()
            conf_var = q_t - max(sig_t, self.eps) * thresh
            conformal_vars.append(conf_var)
            thresholds.append(thresh)

            s_t, _, err_t = self.update(q_t, r_t, sig_t, conformal_var=conf_var)
            violations.append(err_t)
            scaled_residuals.append(s_t)

        return pd.DataFrame(
            {
                "conformal_var": conformal_vars,
                "base_var": base_predictions.to_numpy(),
                "threshold": thresholds,
                "alpha_t": alpha_ts,
                "violation": violations,
                "scaled_residual": scaled_residuals,
                "sigma": sigmas.to_numpy(),
            },
            index=base_predictions.index,
        )
