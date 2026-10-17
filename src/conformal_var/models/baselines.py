"""Institutional Value-at-Risk (VaR) baseline forecasters.

Implements a unified BaseVaRForecaster interface with strictly 1-step-ahead
out-of-sample forecasting (using returns up to t-1 to forecast risk for day t).

Four institutional models:
1. HistoricalSimulationVaR: Empirical quantile of rolling window (e.g. 252 or 500 days).
2. RiskMetricsEWMAVaR: J.P. Morgan EWMA variance (lambda=0.94) with Gaussian innovations.
3. GARCHNormalVaR: GARCH(1,1) with Gaussian innovations, 21-day periodic parameter refit,
   and daily recurrence updates.
4. GARCHStudentTVaR: GARCH(1,1) with Skewed Student-t (or Student-t) innovations to capture
   fat tails and negative asymmetry.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from arch import arch_model
from scipy.stats import norm

logger = logging.getLogger(__name__)


def _to_clean_1d_array(returns: pd.Series | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Extract clean 1D float64 numpy array without NaNs."""
    if isinstance(returns, pd.Series):
        arr: npt.NDArray[np.float64] = returns.dropna().to_numpy(dtype=np.float64)
    else:
        raw_arr = np.asarray(returns, dtype=np.float64)
        arr = raw_arr[~np.isnan(raw_arr)]
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


class BaseVaRForecaster(ABC):
    """Abstract base class for 1-step-ahead out-of-sample VaR forecasting."""

    @abstractmethod
    def fit(self, returns: pd.Series | npt.NDArray[np.float64]) -> BaseVaRForecaster:
        """Fit model on historical returns strictly up to day t-1.

        Parameters
        ----------
        returns : pd.Series | np.ndarray
            Past log returns series r_0, r_1, ..., r_{t-1}.

        Returns
        -------
        BaseVaRForecaster
            Fitted forecaster instance ready to predict VaR for day t.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_var(self, alpha: float) -> float:
        """Predict 1-step-ahead out-of-sample VaR for day t at quantile alpha.

        Parameters
        ----------
        alpha : float
            Risk quantile level (e.g. 0.05 for 95% VaR, 0.01 for 99% VaR).

        Returns
        -------
        float
            Forecasted return quantile (typically negative, representing downside threshold).
        """
        raise NotImplementedError

    def predict_rolling(
        self,
        returns: pd.Series,
        alphas: Sequence[float] = (0.05, 0.01),
        min_history: int = 252,
    ) -> pd.DataFrame:
        """Compute sequential out-of-sample 1-step-ahead VaR across a series.

        At each trading day t, the forecast uses strictly data up to t-1.

        Parameters
        ----------
        returns : pd.Series
            Datetime-indexed return series.
        alphas : Sequence[float]
            Quantile levels (e.g. [0.05, 0.01]).
        min_history : int
            Initial warm-up window size.

        Returns
        -------
        pd.DataFrame
            Out-of-sample 1-step-ahead VaR forecasts for each alpha.
        """
        clean_s = returns.dropna()
        n = len(clean_s)
        if n <= min_history:
            raise ValueError(
                f"Return series length ({n}) must exceed min_history ({min_history})"
            )

        col_names = [f"VaR_{int((1.0 - a) * 100)}" for a in alphas]
        records: list[dict[str, float]] = []
        valid_indices: list[Any] = []

        for i in range(min_history, n):
            hist_slice = clean_s.iloc[:i]
            self.fit(hist_slice)
            row_dict = {
                col_name: self.predict_var(alpha)
                for alpha, col_name in zip(alphas, col_names, strict=True)
            }
            records.append(row_dict)
            valid_indices.append(clean_s.index[i])

        return pd.DataFrame(records, index=valid_indices)


class HistoricalSimulationVaR(BaseVaRForecaster):
    """Historical Simulation VaR using empirical quantile of rolling past returns."""

    def __init__(self, window: int = 252) -> None:
        if window < 10:
            raise ValueError(f"Historical window must be >= 10, got {window}")
        self.window = window
        self._window_returns: npt.NDArray[np.float64] | None = None

    def fit(self, returns: pd.Series | npt.NDArray[np.float64]) -> HistoricalSimulationVaR:
        """Fit by retaining the most recent window of returns up to t-1."""
        arr = _to_clean_1d_array(returns)
        if len(arr) < self.window:
            raise ValueError(
                f"HistoricalSimulationVaR requires at least {self.window} returns, got {len(arr)}"
            )
        self._window_returns = arr[-self.window:]
        return self

    def predict_var(self, alpha: float) -> float:
        """Compute empirical alpha-quantile of the past window."""
        if self._window_returns is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Quantile alpha must be in (0, 1), got {alpha}")
        return float(np.quantile(self._window_returns, alpha, method="linear"))


class RiskMetricsEWMAVaR(BaseVaRForecaster):
    """J.P. Morgan RiskMetrics EWMA volatility model with decay factor lambda=0.94."""

    def __init__(self, decay_factor: float = 0.94) -> None:
        if not (0.0 < decay_factor < 1.0):
            raise ValueError(f"decay_factor lambda must be in (0, 1), got {decay_factor}")
        self.decay_factor = decay_factor
        self._forecast_sigma: float | None = None

    def fit(self, returns: pd.Series | npt.NDArray[np.float64]) -> RiskMetricsEWMAVaR:
        """Compute 1-step-ahead conditional volatility sigma_t recursively from r_0...r_{t-1}."""
        arr = _to_clean_1d_array(returns)
        n = len(arr)
        if n < 2:
            raise ValueError(f"RiskMetrics EWMA requires at least 2 returns, got {n}")

        # Initialize variance with first return squared or initial sample variance
        lam = self.decay_factor
        var_t = float(arr[0] ** 2)

        for k in range(1, n):
            var_t = lam * var_t + (1.0 - lam) * float(arr[k] ** 2)

        # 1-step-ahead forecast for day t using data up to t-1:
        # sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
        # Since the loop updated up to arr[n-1], var_t is already the forecast for day t!
        self._forecast_sigma = np.sqrt(max(var_t, 1e-12))
        return self

    def predict_var(self, alpha: float) -> float:
        """Compute VaR_t(alpha) = Phi^{-1}(alpha) * sigma_t."""
        if self._forecast_sigma is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Quantile alpha must be in (0, 1), got {alpha}")
        z_alpha = float(norm.ppf(alpha))
        return z_alpha * self._forecast_sigma


class GARCHNormalVaR(BaseVaRForecaster):
    """GARCH(1,1) with Gaussian innovations.

    Refits parameters every 21 trading days over a rolling window (default 756 days),
    updating conditional variance daily via:
        sigma_t^2 = omega + alpha_1 r_{t-1}^2 + beta_1 sigma_{t-1}^2.
    Rescales returns by 100x internally for optimizer stability,
    scaling forecasts back to raw units.
    """

    def __init__(
        self,
        training_window: int = 756,
        refit_every: int = 21,
        mean: Literal["Zero", "Constant"] = "Zero",
    ) -> None:
        if training_window < 100:
            raise ValueError(f"training_window must be >= 100, got {training_window}")
        if refit_every < 1:
            raise ValueError(f"refit_every must be >= 1, got {refit_every}")
        self.training_window = training_window
        self.refit_every = refit_every
        self.mean = mean

        # Cached estimation state
        self._last_fitted_n: int = -1
        self._omega: float = 0.0
        self._alpha1: float = 0.0
        self._beta1: float = 0.0
        self._last_cond_var_scaled: float = 0.0
        self._forecast_sigma: float | None = None

    def _refit(self, window_arr_scaled: npt.NDArray[np.float64]) -> None:
        """Full maximum likelihood estimation of GARCH(1,1) on 100x rescaled returns."""
        am = arch_model(
            window_arr_scaled,
            mean=self.mean,
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
        )
        res = am.fit(disp="off", show_warning=False)

        self._omega = float(res.params.get("omega", 0.01))
        self._alpha1 = float(res.params.get("alpha[1]", 0.05))
        self._beta1 = float(res.params.get("beta[1]", 0.90))

        # 1-step-ahead forecast of conditional variance in scaled units
        forecast_res = res.forecast(horizon=1)
        self._last_cond_var_scaled = float(forecast_res.variance.iloc[-1, 0])

    def fit(self, returns: pd.Series | npt.NDArray[np.float64]) -> GARCHNormalVaR:
        """Fit or recursively update conditional volatility for day t using returns up to t-1."""
        arr = _to_clean_1d_array(returns)
        n = len(arr)
        if n < self.training_window:
            raise ValueError(
                f"GARCHNormalVaR requires at least {self.training_window} returns, got {n}"
            )

        # Rescale returns by 100x for optimizer numerical stability
        window_arr = arr[-self.training_window:]
        window_scaled = window_arr * 100.0

        # Check if periodic refit is triggered
        needs_refit = (
            self._last_fitted_n < 0
            or (n - self._last_fitted_n) >= self.refit_every
            or self._last_cond_var_scaled <= 0.0
        )

        if needs_refit:
            self._refit(window_scaled)
            self._last_fitted_n = n
        else:
            # Daily recurrence update: sigma_t^2 = omega + alpha_1 r_{t-1}^2 + beta_1 sigma_{t-1}^2
            r_prev_scaled = window_scaled[-1]
            self._last_cond_var_scaled = (
                self._omega
                + self._alpha1 * (r_prev_scaled ** 2)
                + self._beta1 * self._last_cond_var_scaled
            )

        # Scale forecasted standard deviation back to raw log-return units
        sigma_scaled = np.sqrt(max(self._last_cond_var_scaled, 1e-12))
        self._forecast_sigma = sigma_scaled / 100.0
        return self

    def predict_var(self, alpha: float) -> float:
        """Predict 1-step-ahead Gaussian VaR."""
        if self._forecast_sigma is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Quantile alpha must be in (0, 1), got {alpha}")
        z_alpha = float(norm.ppf(alpha))
        return z_alpha * self._forecast_sigma


class GARCHStudentTVaR(BaseVaRForecaster):
    """GARCH(1,1) with Skewed Student-t (or Student-t) innovations.

    Captures fat tails and negative return skewness. Refits parameters every 21 trading
    days on a rolling 756-day window, updating conditional variance daily via GARCH recurrence.
    """

    def __init__(
        self,
        training_window: int = 756,
        refit_every: int = 21,
        dist: Literal["skewt", "t"] = "skewt",
        mean: Literal["Zero", "Constant"] = "Zero",
    ) -> None:
        if training_window < 100:
            raise ValueError(f"training_window must be >= 100, got {training_window}")
        if refit_every < 1:
            raise ValueError(f"refit_every must be >= 1, got {refit_every}")
        if dist not in ("skewt", "t"):
            raise ValueError(f"dist must be 'skewt' or 't', got {dist}")
        self.training_window = training_window
        self.refit_every = refit_every
        self.dist = dist
        self.mean = mean

        # Cached estimation state
        self._last_fitted_n: int = -1
        self._omega: float = 0.0
        self._alpha1: float = 0.0
        self._beta1: float = 0.0
        self._dist_params: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self._distribution: Any = None
        self._last_cond_var_scaled: float = 0.0
        self._forecast_sigma: float | None = None

    def _refit(self, window_arr_scaled: npt.NDArray[np.float64]) -> None:
        """Full maximum likelihood estimation of GARCH(1,1) with fat-tailed innovations."""
        try:
            am = arch_model(
                window_arr_scaled,
                mean=self.mean,
                vol="GARCH",
                p=1,
                q=1,
                dist=self.dist,
            )
            res = am.fit(disp="off", show_warning=False)
        except Exception as exc:
            # Fall back gracefully to Student-t if skewt numerical optimization fails
            logger.warning(
                "Optimization failed for dist='%s' (%s), falling back to 't'",
                self.dist,
                exc,
            )
            am = arch_model(
                window_arr_scaled,
                mean=self.mean,
                vol="GARCH",
                p=1,
                q=1,
                dist="t",
            )
            res = am.fit(disp="off", show_warning=False)

        self._omega = float(res.params.get("omega", 0.01))
        self._alpha1 = float(res.params.get("alpha[1]", 0.05))
        self._beta1 = float(res.params.get("beta[1]", 0.90))

        # Extract distribution object and parameters
        self._distribution = am.distribution
        dist_param_names = am.distribution.parameter_names()
        self._dist_params = res.params[dist_param_names].to_numpy(dtype=np.float64)

        # 1-step-ahead forecast in scaled units
        forecast_res = res.forecast(horizon=1)
        self._last_cond_var_scaled = float(forecast_res.variance.iloc[-1, 0])

    def fit(self, returns: pd.Series | npt.NDArray[np.float64]) -> GARCHStudentTVaR:
        """Fit or recursively update conditional volatility for day t using returns up to t-1."""
        arr = _to_clean_1d_array(returns)
        n = len(arr)
        if n < self.training_window:
            raise ValueError(
                f"GARCHStudentTVaR requires at least {self.training_window} returns, got {n}"
            )

        window_arr = arr[-self.training_window:]
        window_scaled = window_arr * 100.0

        needs_refit = (
            self._last_fitted_n < 0
            or (n - self._last_fitted_n) >= self.refit_every
            or self._last_cond_var_scaled <= 0.0
        )

        if needs_refit:
            self._refit(window_scaled)
            self._last_fitted_n = n
        else:
            # Daily recurrence update
            r_prev_scaled = window_scaled[-1]
            self._last_cond_var_scaled = (
                self._omega
                + self._alpha1 * (r_prev_scaled ** 2)
                + self._beta1 * self._last_cond_var_scaled
            )

        sigma_scaled = np.sqrt(max(self._last_cond_var_scaled, 1e-12))
        self._forecast_sigma = sigma_scaled / 100.0
        return self

    def predict_var(self, alpha: float) -> float:
        """Predict 1-step-ahead fat-tailed VaR using fitted distribution quantile."""
        if self._forecast_sigma is None or self._distribution is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Quantile alpha must be in (0, 1), got {alpha}")

        q_alpha = float(self._distribution.ppf(alpha, self._dist_params))
        return q_alpha * self._forecast_sigma
