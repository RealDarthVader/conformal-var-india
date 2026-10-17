"""LightGBM Quantile Regression VaR engine with walk-forward retraining.

Trains separate LGBMRegressor models for target risk quantiles (e.g. alpha in {0.01, 0.05})
with strictly regularized hyperparameters to prevent overfitting noisy financial returns.
Supports walk-forward retraining every 63 trading days on a rolling 756-day window,
reserving the most recent 252 trading days as an untouched calibration/validation buffer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


class LightGBMQuantileVaR:
    """LightGBM Quantile Regressor for Value-at-Risk forecasting."""

    def __init__(
        self,
        alphas: Sequence[float] = (0.05, 0.01),
        training_window: int = 756,
        calibration_buffer: int = 252,
        refit_every: int = 63,
        max_depth: int = 4,
        num_leaves: int = 15,
        min_child_samples: int = 30,
        learning_rate: float = 0.03,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        for a in alphas:
            if not (0.0 < a < 1.0):
                raise ValueError(f"Quantile alpha must be in (0, 1), got {a}")
        if calibration_buffer >= training_window:
            msg = (
                f"calibration_buffer ({calibration_buffer}) "
                f"must be < training_window ({training_window})"
            )
            raise ValueError(msg)

        self.alphas = tuple(sorted(alphas, reverse=True))
        self.training_window = training_window
        self.calibration_buffer = calibration_buffer
        self.refit_every = refit_every
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._models: dict[float, lgb.LGBMRegressor] = {}
        self._last_calibration_X: pd.DataFrame | None = None
        self._last_calibration_y: pd.Series | None = None

    def _build_lgbm(self, alpha: float) -> lgb.LGBMRegressor:
        """Instantiate a regularized LightGBM quantile regressor."""
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1,
        )

    def fit(
        self,
        X: pd.DataFrame | npt.NDArray[np.float64],
        y: pd.Series | npt.NDArray[np.float64],
        alphas: Sequence[float] | None = None,
    ) -> LightGBMQuantileVaR:
        """Fit quantile regressors for specified alphas on training data.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix (m_samples, n_features).
        y : pd.Series | np.ndarray
            Target return vector (m_samples,).
        alphas : Sequence[float] | None
            Quantile levels to fit (defaults to self.alphas).

        Returns
        -------
        LightGBMQuantileVaR
            Fitted forecaster instance.
        """
        fit_alphas = self.alphas if alphas is None else tuple(alphas)

        # Convert to clean arrays/DataFrames
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_ser = pd.Series(y) if not isinstance(y, pd.Series) else y

        # Remove any rows with NaN in features or target
        valid_mask = ~(X_df.isna().any(axis=1) | y_ser.isna())
        clean_X = X_df[valid_mask]
        clean_y = y_ser[valid_mask]

        if len(clean_X) < self.min_child_samples:
            raise ValueError(
                f"Insufficient training observations: {len(clean_X)} < {self.min_child_samples}"
            )

        for alpha in fit_alphas:
            model = self._build_lgbm(alpha)
            model.fit(clean_X, clean_y)
            self._models[alpha] = model

        return self

    def predict_var(
        self,
        X: pd.DataFrame | pd.Series | npt.NDArray[np.float64],
        alpha: float = 0.05,
    ) -> float:
        """Predict 1-step-ahead VaR at quantile level alpha for a single feature vector.

        Parameters
        ----------
        X : pd.DataFrame | pd.Series | np.ndarray
            Feature vector for day t (known at close of day t-1).
        alpha : float
            Quantile level.

        Returns
        -------
        float
            Predicted quantile (typically negative).
        """
        if alpha not in self._models:
            fitted = list(self._models.keys())
            raise KeyError(
                f"Model for alpha={alpha} has not been trained. Fitted alphas: {fitted}"
            )

        if isinstance(X, pd.Series):
            x_input = X.to_frame().T
        elif isinstance(X, np.ndarray):
            x_input = X.reshape(1, -1) if X.ndim == 1 else X
        else:
            x_input = X.iloc[[0]] if len(X) > 1 else X

        pred = self._models[alpha].predict(x_input)
        return float(pred[0])

    def predict_rolling(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        alphas: Sequence[float] | None = None,
    ) -> pd.DataFrame:
        """Execute walk-forward out-of-sample forecasting with calibration hold-out.

        Walks forward day by day from index `training_window` onward:
        - At day t, considers the rolling `training_window` history up to t-1.
        - Holds out the most recent `calibration_buffer` days as untouched validation data.
        - Retrains LightGBM every `refit_every` days on the earlier portion of the window.
        - Predicts 1-step-ahead VaR for day t using row X.loc[t].

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (pre-shifted by 1).
        y : pd.Series
            Target return series.
        alphas : Sequence[float] | None
            Quantiles to predict.

        Returns
        -------
        pd.DataFrame
            Out-of-sample 1-step-ahead VaR predictions.
        """
        eval_alphas = self.alphas if alphas is None else tuple(alphas)
        n = len(X)
        if n < self.training_window:
            raise ValueError(
                f"Total observations ({n}) must be >= training_window ({self.training_window})"
            )

        col_names = [f"VaR_{int((1.0 - a) * 100)}" for a in eval_alphas]
        records: list[dict[str, float]] = []
        valid_indices: list[Any] = []

        last_refit_idx: int = -999

        for t in range(self.training_window, n):
            # Window of data up to t-1
            window_start = t - self.training_window
            window_end = t  # exclusive

            # Check if walk-forward retraining is triggered
            if (t - last_refit_idx) >= self.refit_every or not self._models:
                # Reserve recent calibration_buffer days untouched
                train_end = window_end - self.calibration_buffer
                X_train = X.iloc[window_start:train_end]
                y_train = y.iloc[window_start:train_end]

                # Update calibration buffer slice
                self._last_calibration_X = X.iloc[train_end:window_end]
                self._last_calibration_y = y.iloc[train_end:window_end]

                self.fit(X_train, y_train, alphas=eval_alphas)
                last_refit_idx = t

            # Predict 1-step-ahead VaR for day t using X_t
            x_t = X.iloc[[t]]
            row_dict: dict[str, float] = {}
            for alpha, col_name in zip(eval_alphas, col_names, strict=True):
                pred_val = float(self._models[alpha].predict(x_t)[0])
                row_dict[col_name] = pred_val

            records.append(row_dict)
            valid_indices.append(X.index[t])

        return pd.DataFrame(records, index=valid_indices)

    @property
    def calibration_buffer_data(self) -> tuple[pd.DataFrame, pd.Series] | None:
        """Return the most recent untouched calibration buffer (X_cal, y_cal)."""
        if self._last_calibration_X is None or self._last_calibration_y is None:
            return None
        return self._last_calibration_X, self._last_calibration_y
