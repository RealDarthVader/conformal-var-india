"""Causal feature engineering pipeline for Indian equity VaR forecasting.

Constructs feature matrix X_t for each asset where every feature at index t
is computed STRICTLY from information available at or before market close of day t-1.
All features are shifted by 1 day to guarantee zero lookahead data leakage.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

# Feature column constants
FEATURE_NAMES: Final[list[str]] = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "realized_vol_5d",
    "realized_vol_10d",
    "realized_vol_21d",
    "realized_vol_63d",
    "vol_term_structure",
    "parkinson_vol_5d",
    "parkinson_vol_21d",
    "downside_semi_dev_21d",
    "skew_21d",
    "kurt_63d",
    "max_drawdown_21d",
    "market_return_5d",
    "vix_level",
    "vix_change_5d",
]


def compute_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling Parkinson High-Low intraday volatility estimator.

    sigma_Parkinson = sqrt( 1 / (4 * ln(2) * window) * sum( (ln(H/L))^2 ) )
    """
    ratio = np.maximum(high / np.maximum(low, 1e-8), 1.0)
    log_hl = np.log(ratio)
    factor = 1.0 / (4.0 * np.log(2.0))
    squared_terms = factor * (log_hl ** 2)
    rolling_var = squared_terms.rolling(window=window, min_periods=window).mean()
    return pd.Series(np.sqrt(np.maximum(rolling_var, 0.0)), index=high.index)


def compute_downside_semi_deviation(
    returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Compute rolling downside semi-deviation (volatility of negative returns only).

    semi_sigma = sqrt( 1 / window * sum( min(r_k, 0)^2 ) )
    """
    negative_returns = np.minimum(returns, 0.0)
    squared_neg = negative_returns ** 2
    rolling_semi_var = squared_neg.rolling(window=window, min_periods=window).mean()
    return pd.Series(np.sqrt(np.maximum(rolling_semi_var, 0.0)), index=returns.index)


def compute_rolling_max_drawdown(
    returns: pd.Series,
    window: int = 21,
) -> pd.Series:
    """Compute rolling maximum drawdown over a trailing window of log returns."""
    # Convert log returns to cumulative wealth relative to window start
    def _window_max_dd(window_slice: npt.NDArray[np.float64]) -> float:
        cum_ret = np.cumsum(window_slice)
        wealth = np.exp(cum_ret)
        running_max = np.maximum.accumulate(wealth)
        dd = (wealth - running_max) / running_max
        return float(np.min(dd))

    # Apply rolling window
    return returns.rolling(window=window, min_periods=window).apply(
        _window_max_dd,
        raw=True,
    )


class FeatureEngineer:
    """Engine for generating causally-aligned feature matrices for asset returns."""

    def __init__(
        self,
        market_benchmark: str = "^NSEI",
        vix_benchmark: str = "^INDIAVIX",
    ) -> None:
        self.market_benchmark = market_benchmark
        self.vix_benchmark = vix_benchmark

    def compute_features(
        self,
        asset_returns: pd.Series,
        asset_high: pd.Series | None = None,
        asset_low: pd.Series | None = None,
        market_returns: pd.Series | None = None,
        vix_prices: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Compute feature matrix X_t strictly shifted by 1 to prevent lookahead leakage.

        Parameters
        ----------
        asset_returns : pd.Series
            Daily log returns for target asset.
        asset_high : pd.Series | None
            Daily High prices for target asset (optional, falls back if missing).
        asset_low : pd.Series | None
            Daily Low prices for target asset (optional, falls back if missing).
        market_returns : pd.Series | None
            Daily log returns for market benchmark (^NSEI).
        vix_prices : pd.Series | None
            Daily level of India VIX (^INDIAVIX).

        Returns
        -------
        pd.DataFrame
            Feature matrix with index aligned to asset_returns, where row t
            depends strictly on information from t-1 and earlier.
        """
        r = asset_returns.astype(np.float64)
        features: dict[str, pd.Series] = {}

        # 1. Lagged log returns: r_{t-1}, r_{t-2}, r_{t-3}, r_{t-5}
        features["return_lag_1"] = r.shift(1)
        features["return_lag_2"] = r.shift(2)
        features["return_lag_3"] = r.shift(3)
        features["return_lag_5"] = r.shift(5)

        # 2. Multi-horizon realized volatility (rolling std over 5, 10, 21, 63 days)
        # Shifted by 1 so date t only sees std up to day t-1
        vol_5d_raw = r.rolling(5, min_periods=5).std()
        vol_10d_raw = r.rolling(10, min_periods=10).std()
        vol_21d_raw = r.rolling(21, min_periods=21).std()
        vol_63d_raw = r.rolling(63, min_periods=63).std()

        features["realized_vol_5d"] = vol_5d_raw.shift(1)
        features["realized_vol_10d"] = vol_10d_raw.shift(1)
        features["realized_vol_21d"] = vol_21d_raw.shift(1)
        features["realized_vol_63d"] = vol_63d_raw.shift(1)

        # Volatility term structure ratio: sigma_5d / sigma_63d
        term_structure = vol_5d_raw / (vol_63d_raw + 1e-8)
        features["vol_term_structure"] = term_structure.shift(1)

        # 3. Parkinson High-Low intraday volatility estimator (rolling 5-day and 21-day)
        if asset_high is not None and asset_low is not None:
            p_vol_5d_raw = compute_parkinson_volatility(asset_high, asset_low, window=5)
            p_vol_21d_raw = compute_parkinson_volatility(asset_high, asset_low, window=21)
            features["parkinson_vol_5d"] = p_vol_5d_raw.shift(1)
            features["parkinson_vol_21d"] = p_vol_21d_raw.shift(1)
        else:
            # Fallback to realized vol proxy if intraday high/low not provided
            features["parkinson_vol_5d"] = features["realized_vol_5d"]
            features["parkinson_vol_21d"] = features["realized_vol_21d"]

        # 4. Downside semi-deviation (rolling 21-day volatility of negative returns only)
        semi_dev_21d_raw = compute_downside_semi_deviation(r, window=21)
        features["downside_semi_dev_21d"] = semi_dev_21d_raw.shift(1)

        # 5. Higher moments & drawdown: rolling 21d skew, rolling 63d excess kurtosis, 21d max dd
        features["skew_21d"] = r.rolling(21, min_periods=21).skew().shift(1)
        features["kurt_63d"] = r.rolling(63, min_periods=63).kurt().shift(1)
        max_dd_21d_raw = compute_rolling_max_drawdown(r, window=21)
        features["max_drawdown_21d"] = max_dd_21d_raw.shift(1)

        # 6. Market context: lagged ^NSEI 5-day return, lagged ^INDIAVIX level, 5-day VIX change
        if market_returns is not None:
            m_ret_5d_raw = market_returns.rolling(5, min_periods=5).sum()
            features["market_return_5d"] = m_ret_5d_raw.shift(1)
        else:
            features["market_return_5d"] = features["return_lag_1"]

        if vix_prices is not None:
            vix_clean = vix_prices.astype(np.float64)
            features["vix_level"] = vix_clean.shift(1)
            vix_change_raw = vix_clean - vix_clean.shift(5)
            features["vix_change_5d"] = vix_change_raw.shift(1)
        else:
            features["vix_level"] = pd.Series(20.0, index=r.index)
            features["vix_change_5d"] = pd.Series(0.0, index=r.index)

        df_features = pd.DataFrame(features, index=r.index)
        return df_features[FEATURE_NAMES]

    def create_dataset_from_cache(
        self,
        target_ticker: str,
        returns_df: pd.DataFrame,
        ohlcv_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract features and target returns from cached dataframes.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            (X, y) aligned feature matrix and target return series.
        """
        asset_returns = returns_df[target_ticker]

        # Extract High and Low if multi-index OHLCV
        asset_high: pd.Series | None = None
        asset_low: pd.Series | None = None
        vix_prices: pd.Series | None = None

        if isinstance(ohlcv_df.columns, pd.MultiIndex):
            if "High" in ohlcv_df.columns and target_ticker in ohlcv_df["High"].columns:
                asset_high = ohlcv_df["High"][target_ticker]
            if "Low" in ohlcv_df.columns and target_ticker in ohlcv_df["Low"].columns:
                asset_low = ohlcv_df["Low"][target_ticker]
            if "Close" in ohlcv_df.columns and self.vix_benchmark in ohlcv_df["Close"].columns:
                vix_prices = ohlcv_df["Close"][self.vix_benchmark]
        elif target_ticker in ohlcv_df.columns:
            if "High" in ohlcv_df:
                asset_high = ohlcv_df["High"]
            if "Low" in ohlcv_df:
                asset_low = ohlcv_df["Low"]

        market_returns: pd.Series | None = None
        if self.market_benchmark in returns_df.columns:
            market_returns = returns_df[self.market_benchmark]

        X = self.compute_features(
            asset_returns=asset_returns,
            asset_high=asset_high,
            asset_low=asset_low,
            market_returns=market_returns,
            vix_prices=vix_prices,
        )

        y = asset_returns.copy()
        y.name = "target"
        return X, y
