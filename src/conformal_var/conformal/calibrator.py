"""Conformal calibration engines for one-sided lower-tail Value-at-Risk (VaR).

Wraps any base quantile forecaster q_alpha(X_t) with finite-sample coverage guarantees.
Implements:
1. SplitConformalVaR: Static calibration over M=252 days with finite-sample quantile level.
2. SlidingWindowConformalVaR: Rolling calibration over the most recent M=252 residuals.
3. AdaptiveConformalVaR (Gibbs & Candès, 2021 - ACI): Online dynamic learning rate
   updating alpha_t to track nominal coverage under non-stationarity and volatility shocks.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


def compute_conformity_residual(
    base_prediction: float | npt.NDArray[np.float64] | pd.Series,
    realized_return: float | npt.NDArray[np.float64] | pd.Series,
) -> Any:
    """Compute signed conformity residual where a VaR violation corresponds to s_t > 0.

    s_t = q_hat_alpha(X_t) - r_t
    """
    return base_prediction - realized_return


def finite_sample_conformal_quantile(
    residuals: npt.NDArray[np.float64] | Sequence[float],
    alpha: float,
) -> float:
    """Compute the finite-sample adjusted conformal threshold Q_{1-alpha}.

    level = ceil((M + 1) * (1 - alpha)) / M
    """
    arr = np.asarray(residuals, dtype=np.float64)
    m = len(arr)
    if m == 0:
        return 0.0

    raw_level = math.ceil((m + 1) * (1.0 - alpha)) / m
    level = min(max(raw_level, 0.0), 1.0)
    return float(np.quantile(arr, level, method="linear"))


class SplitConformalVaR:
    """Split Conformal Prediction using a static calibration buffer.

    Computes Q_{1-alpha} over a fixed calibration set of size M=252 using
    the finite-sample quantile level ceil((M+1)(1-alpha))/M.
    """

    def __init__(self, alpha: float = 0.05, calibration_size: int = 252) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.calibration_size = calibration_size
        self._q_hat: float | None = None
        self._residuals: npt.NDArray[np.float64] | None = None

    @property
    def q_hat(self) -> float:
        """Calibrated offset threshold Q_{1-alpha}."""
        if self._q_hat is None:
            raise RuntimeError("Model has not been calibrated. Call calibrate() first.")
        return self._q_hat

    def calibrate(
        self,
        base_predictions: pd.Series | npt.NDArray[np.float64],
        returns: pd.Series | npt.NDArray[np.float64],
    ) -> SplitConformalVaR:
        """Calibrate threshold on holdout validation data."""
        q_arr = np.asarray(base_predictions, dtype=np.float64).ravel()
        r_arr = np.asarray(returns, dtype=np.float64).ravel()

        if len(q_arr) != len(r_arr):
            raise ValueError(
                f"Length mismatch: predictions ({len(q_arr)}) vs returns ({len(r_arr)})"
            )
        if len(q_arr) < 10:
            raise ValueError(f"Calibration set too small: {len(q_arr)} observations")

        residuals = compute_conformity_residual(q_arr, r_arr)
        self._residuals = residuals
        self._q_hat = finite_sample_conformal_quantile(residuals, self.alpha)
        return self

    def predict_var(self, base_prediction: float) -> float:
        """Compute conformalized VaR: VaR_t = q_hat(X_t) - Q_{1-alpha}."""
        return base_prediction - self.q_hat

    def predict(
        self,
        base_predictions: pd.Series | npt.NDArray[np.float64],
    ) -> pd.Series | npt.NDArray[np.float64]:
        """Compute conformalized VaR for an array or Series of base predictions."""
        if isinstance(base_predictions, pd.Series):
            return base_predictions - self.q_hat
        arr = np.asarray(base_predictions, dtype=np.float64)
        return arr - self.q_hat


class SlidingWindowConformalVaR:
    """Sliding-window Conformal VaR over the most recent M=252 residuals."""

    def __init__(self, alpha: float = 0.05, window: int = 252) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if window < 10:
            raise ValueError(f"window must be >= 10, got {window}")
        self.alpha = alpha
        self.window = window
        self._residual_buffer: collections.deque[float] = collections.deque(maxlen=window)

    def calibrate(
        self,
        base_predictions: pd.Series | npt.NDArray[np.float64],
        returns: pd.Series | npt.NDArray[np.float64],
    ) -> SlidingWindowConformalVaR:
        """Initialize the sliding window buffer with calibration residuals."""
        q_arr = np.asarray(base_predictions, dtype=np.float64).ravel()
        r_arr = np.asarray(returns, dtype=np.float64).ravel()

        residuals = compute_conformity_residual(q_arr, r_arr)
        self._residual_buffer.clear()
        for res in residuals[-self.window:]:
            self._residual_buffer.append(float(res))
        return self

    def current_threshold(self) -> float:
        """Compute Q_{1-alpha} over current residual window."""
        if not self._residual_buffer:
            return 0.0
        return finite_sample_conformal_quantile(list(self._residual_buffer), self.alpha)

    def predict_var(self, base_prediction: float) -> float:
        """Predict 1-step-ahead conformalized VaR for day t."""
        return base_prediction - self.current_threshold()

    def update(self, base_prediction: float, realized_return: float) -> float:
        """Incorporate realized return at market close of day t to update window for t+1."""
        res = float(compute_conformity_residual(base_prediction, realized_return))
        self._residual_buffer.append(res)
        return res

    def predict_sequence(
        self,
        base_predictions: pd.Series,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """Run sequential walk-forward calibration day by day."""
        if len(base_predictions) != len(returns):
            raise ValueError("Length mismatch between predictions and returns")

        conformal_vars: list[float] = []
        thresholds: list[float] = []
        violations: list[int] = []

        for q_t, r_t in zip(base_predictions, returns, strict=True):
            thresh = self.current_threshold()
            conf_var = q_t - thresh
            conformal_vars.append(conf_var)
            thresholds.append(thresh)
            is_violation = int(r_t < conf_var)
            violations.append(is_violation)
            self.update(q_t, r_t)

        return pd.DataFrame(
            {
                "conformal_var": conformal_vars,
                "base_var": base_predictions.to_numpy(),
                "threshold": thresholds,
                "violation": violations,
            },
            index=base_predictions.index,
        )


class AdaptiveConformalVaR:
    """Adaptive Conformal Inference (Gibbs & Candès, 2021 - ACI).

    Dynamically adjusts effective miscoverage rate alpha_t online after each trading day:
    alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t), alpha_min, alpha_max)
    """

    ALLOWED_GAMMAS: Sequence[float] = (0.005, 0.01, 0.025, 0.05)

    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 0.01,
        window: int = 252,
        alpha_min: float = 0.001,
        alpha_max: float = 0.999,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.alpha = alpha
        self.gamma = gamma
        self.window = window
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.alpha_t: float = alpha
        self._residual_buffer: collections.deque[float] = collections.deque(maxlen=window)
        self._last_predicted_var: float | None = None

    def calibrate(
        self,
        base_predictions: pd.Series | npt.NDArray[np.float64],
        returns: pd.Series | npt.NDArray[np.float64],
    ) -> AdaptiveConformalVaR:
        """Initialize the residual buffer with initial validation residuals."""
        q_arr = np.asarray(base_predictions, dtype=np.float64).ravel()
        r_arr = np.asarray(returns, dtype=np.float64).ravel()

        residuals = compute_conformity_residual(q_arr, r_arr)
        self._residual_buffer.clear()
        for res in residuals[-self.window:]:
            self._residual_buffer.append(float(res))
        self.alpha_t = self.alpha
        return self

    def current_threshold(self) -> float:
        """Compute Q_{1 - alpha_t} over recent residuals using current alpha_t."""
        if not self._residual_buffer:
            return 0.0
        return finite_sample_conformal_quantile(list(self._residual_buffer), self.alpha_t)

    def predict_var(self, base_prediction: float) -> float:
        """Predict 1-step-ahead adaptive conformalized VaR for day t."""
        thresh = self.current_threshold()
        pred = base_prediction - thresh
        self._last_predicted_var = pred
        return pred

    def update(
        self,
        base_prediction: float,
        realized_return: float,
        conformal_var: float | None = None,
    ) -> tuple[float, float, int]:
        """Update residual window and adapt effective alpha_{t+1} based on violation err_t.

        Returns
        -------
        tuple[float, float, int]
            (residual s_t, updated alpha_{t+1}, violation err_t)
        """
        c_var = conformal_var if conformal_var is not None else self._last_predicted_var
        if c_var is None:
            c_var = self.predict_var(base_prediction)

        err_t = 1 if realized_return < c_var else 0
        s_t = float(compute_conformity_residual(base_prediction, realized_return))
        self._residual_buffer.append(s_t)

        # ACI gradient update: alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t), min, max)
        new_alpha = self.alpha_t + self.gamma * (self.alpha - float(err_t))
        self.alpha_t = float(np.clip(new_alpha, self.alpha_min, self.alpha_max))

        return s_t, self.alpha_t, err_t

    def predict_sequence(
        self,
        base_predictions: pd.Series,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """Execute full online adaptive conformal calibration across a time series."""
        if len(base_predictions) != len(returns):
            raise ValueError("Length mismatch between predictions and returns")

        conformal_vars: list[float] = []
        alpha_ts: list[float] = []
        thresholds: list[float] = []
        violations: list[int] = []
        residuals: list[float] = []

        for q_t, r_t in zip(base_predictions, returns, strict=True):
            alpha_ts.append(self.alpha_t)
            thresh = self.current_threshold()
            conf_var = q_t - thresh
            conformal_vars.append(conf_var)
            thresholds.append(thresh)

            s_t, _, err_t = self.update(q_t, r_t, conformal_var=conf_var)
            violations.append(err_t)
            residuals.append(s_t)

        return pd.DataFrame(
            {
                "conformal_var": conformal_vars,
                "base_var": base_predictions.to_numpy(),
                "alpha_t": alpha_ts,
                "threshold": thresholds,
                "violation": violations,
                "residual": residuals,
            },
            index=base_predictions.index,
        )
