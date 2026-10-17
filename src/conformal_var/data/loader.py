"""Production data loader and preprocessor for Indian equities and indices.

Fetches 15+ years of daily OHLCV and return series from Yahoo Finance,
aligns strictly to the NSE trading calendar (^NSEI), applies forward-fill
limits (max 3 consecutive days, preventing interpolation over long suspensions),
detects corporate action split anomalies, and caches processed datasets
to Parquet files.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Market Benchmark & Volatility Index
MARKET_BENCHMARK: Final[str] = "^NSEI"
VOLATILITY_BENCHMARK: Final[str] = "^INDIAVIX"

# Core Liquid Nifty 50 Constituents
CORE_LIQUID_CONSTITUENTS: Final[list[str]] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "LT.NS",
    "KOTAKBANK.NS",
]

# Stressed / Crashed Historical Constituents (mitigate survivorship bias & capture tail risks)
STRESSED_CONSTITUENTS: Final[list[str]] = [
    "YESBANK.NS",
    "IDEA.NS",
    "ZEEL.NS",
]

# Full default equity risk universe
DEFAULT_UNIVERSE: Final[list[str]] = [
    MARKET_BENCHMARK,
    VOLATILITY_BENCHMARK,
    *CORE_LIQUID_CONSTITUENTS,
    *STRESSED_CONSTITUENTS,
]


@dataclass(frozen=True)
class AnomalyReport:
    """Diagnostic record of an unadjusted corporate action or anomalous return jump."""

    ticker: str
    date: str
    log_return: float
    unadjusted_return: float
    volume: float
    flag_reason: str


class NSEDataLoader:
    """Engine for fetching, cleaning, validating, and caching Indian equities data."""

    def __init__(
        self,
        tickers: Sequence[str] = DEFAULT_UNIVERSE,
        benchmark_ticker: str = MARKET_BENCHMARK,
        start_date: str = "2010-01-01",
        end_date: str = "2026-10-01",
        max_ffill_days: int = 3,
        anomaly_threshold: float = 0.40,
    ) -> None:
        self.tickers = list(tickers)
        self.benchmark_ticker = benchmark_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.max_ffill_days = max_ffill_days
        self.anomaly_threshold = anomaly_threshold

    def fetch_raw_ohlcv(self) -> pd.DataFrame:
        """Download raw multi-ticker OHLCV daily data using yfinance."""
        logger.info(
            "Fetching daily OHLCV for %d tickers from %s to %s",
            len(self.tickers),
            self.start_date,
            self.end_date,
        )
        # Download data with auto_adjust=False to retain unadjusted Close and Adj Close
        raw_df = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            group_by="column",
            auto_adjust=False,
            progress=False,
        )

        if raw_df.empty:
            raise ValueError(f"No market data returned for tickers: {self.tickers}")

        # Normalize index to timezone-naive datetime normalized to midnight
        raw_df.index = pd.to_datetime(raw_df.index).tz_localize(None).normalize()
        raw_df.index.name = "Date"
        return raw_df

    def clean_and_align_ohlcv(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Align data strictly to the NSE calendar and apply forward-fill rules.

        Rules:
        1. Trading calendar is defined strictly by dates when the market benchmark traded.
        2. Filter out non-trading days (weekends, exchange holidays).
        3. Forward-fill missing price series up to a maximum of 3 consecutive trading days.
           Prolonged suspensions (> 3 days) remain un-interpolated (NaN).
        4. Tickers are not backfilled prior to their inception.
        """
        # Ensure benchmark is present
        has_close_tuple = ("Close", self.benchmark_ticker) in raw_df.columns
        has_col = self.benchmark_ticker in raw_df.columns
        if not has_close_tuple and not has_col:
            raise KeyError(
                f"Benchmark ticker '{self.benchmark_ticker}' missing from downloaded dataset."
            )

        # Identify benchmark trading dates where benchmark price exists and is non-null
        if isinstance(raw_df.columns, pd.MultiIndex):
            benchmark_series = raw_df["Close"][self.benchmark_ticker]
        else:
            benchmark_series = raw_df[self.benchmark_ticker]

        nse_trading_days = benchmark_series.dropna().index

        # Restrict rows strictly to NSE trading days
        aligned_df = raw_df.reindex(nse_trading_days)

        # Apply bounded forward-fill to price columns (Open, High, Low, Close, Adj Close)
        # Up to max_ffill_days (3 consecutive trading days). Volume is filled with 0.
        cleaned_parts: list[pd.DataFrame] = []

        if isinstance(aligned_df.columns, pd.MultiIndex):
            metrics = aligned_df.columns.levels[0]
            for metric in metrics:
                metric_df = aligned_df[metric].copy()
                if metric in ("Open", "High", "Low", "Close", "Adj Close"):
                    # Forward-fill gaps up to 3 days; do not backfill before first valid index
                    metric_df = metric_df.ffill(limit=self.max_ffill_days)
                elif metric == "Volume":
                    metric_df = metric_df.fillna(0.0)
                metric_df.columns = pd.MultiIndex.from_product([[metric], metric_df.columns])
                cleaned_parts.append(metric_df)
            result_df = pd.concat(cleaned_parts, axis=1)
        else:
            result_df = aligned_df.ffill(limit=self.max_ffill_days)

        return result_df

    def detect_corporate_action_anomalies(
        self,
        cleaned_ohlcv: pd.DataFrame,
    ) -> list[AnomalyReport]:
        """Flag single-day unadjusted jumps |r_{i,t}| > anomaly_threshold."""
        anomalies: list[AnomalyReport] = []

        if not isinstance(cleaned_ohlcv.columns, pd.MultiIndex):
            return anomalies

        close_df = cleaned_ohlcv.get("Close")
        adj_close_df = cleaned_ohlcv.get("Adj Close")
        volume_df = cleaned_ohlcv.get("Volume")
        high_df = cleaned_ohlcv.get("High")
        low_df = cleaned_ohlcv.get("Low")

        if close_df is None or adj_close_df is None:
            return anomalies

        for ticker in self.tickers:
            if ticker not in close_df.columns or ticker not in adj_close_df.columns:
                continue

            c_series = close_df[ticker].dropna()
            ac_series = adj_close_df[ticker].dropna()

            if len(c_series) < 2 or len(ac_series) < 2:
                continue

            # Compute log returns on unadjusted and adjusted prices
            unadj_ret = np.log(c_series / c_series.shift(1))
            adj_ret = np.log(ac_series / ac_series.shift(1))

            # Discrepancy where unadjusted return has a massive jump (>40%)
            # while adjusted return is small, or where price jumps without high/low span
            mask = np.abs(unadj_ret) > self.anomaly_threshold

            anomaly_dates = unadj_ret[mask].index
            for dt in anomaly_dates:
                u_ret = float(unadj_ret.loc[dt])
                a_ret = float(adj_ret.loc[dt]) if dt in adj_ret.index else u_ret
                vol = (
                    float(volume_df[ticker].loc[dt])
                    if volume_df is not None and dt in volume_df.index
                    else 0.0
                )

                # Check if high/low ratio confirms intraday move
                hl_confirmed = False
                if high_df is not None and low_df is not None and dt in high_df.index:
                    h_val = float(high_df[ticker].loc[dt])
                    l_val = float(low_df[ticker].loc[dt])
                    if l_val > 0 and (h_val / l_val) > 1.30:
                        hl_confirmed = True

                flag_reason = (
                    "Unadjusted split/bonus jump (|unadj_ret| > 0.40)"
                    if abs(u_ret - a_ret) > 0.20
                    else "Extreme jump without intraday confirmation"
                )

                if not hl_confirmed or abs(u_ret - a_ret) > 0.20:
                    report = AnomalyReport(
                        ticker=ticker,
                        date=dt.strftime("%Y-%m-%d"),
                        log_return=round(a_ret, 5),
                        unadjusted_return=round(u_ret, 5),
                        volume=vol,
                        flag_reason=flag_reason,
                    )
                    anomalies.append(report)
                    logger.warning(
                        "Corporate Action Anomaly Detected: %s on %s | "
                        "Unadj: %.2f%%, Adj: %.2f%% | %s",
                        ticker,
                        report.date,
                        u_ret * 100,
                        a_ret * 100,
                        flag_reason,
                    )

        return anomalies

    def compute_log_returns(self, cleaned_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute daily log returns: r_{i,t} = ln(P_{i,t} / P_{i,t-1}) using Adjusted Close.

        Returns NaN on dates prior to ticker inception or during prolonged trading suspensions.
        """
        if isinstance(cleaned_ohlcv.columns, pd.MultiIndex):
            if "Adj Close" in cleaned_ohlcv.columns.levels[0]:
                prices = cleaned_ohlcv["Adj Close"].copy()
            else:
                prices = cleaned_ohlcv["Close"].copy()
        else:
            prices = cleaned_ohlcv.copy()

        # Compute log returns
        returns = np.log(prices / prices.shift(1))

        # First trading date per ticker is naturally NaN
        returns.index.name = "Date"
        return returns

    def run_pipeline(
        self,
        force_reload: bool = False,
        artifacts_dir: Path | str = "artifacts/data",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the full data ingestion, cleaning, anomaly audit, and caching pipeline.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (returns_df, ohlcv_df)
        """
        out_dir = Path(artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        returns_path = out_dir / "returns.parquet"
        ohlcv_path = out_dir / "ohlcv.parquet"

        if not force_reload and returns_path.exists() and ohlcv_path.exists():
            logger.info("Loading cached datasets from %s", out_dir)
            returns_df = pd.read_parquet(returns_path)
            ohlcv_df = pd.read_parquet(ohlcv_path)
            return returns_df, ohlcv_df

        raw_df = self.fetch_raw_ohlcv()
        cleaned_ohlcv = self.clean_and_align_ohlcv(raw_df)
        _anomalies = self.detect_corporate_action_anomalies(cleaned_ohlcv)
        returns_df = self.compute_log_returns(cleaned_ohlcv)

        # Cache to Parquet
        logger.info("Saving cleaned datasets to %s", out_dir)
        cleaned_ohlcv.to_parquet(ohlcv_path, engine="pyarrow")
        returns_df.to_parquet(returns_path, engine="pyarrow")

        return returns_df, cleaned_ohlcv


def load_cached_returns(artifacts_dir: Path | str = "artifacts/data") -> pd.DataFrame:
    """Load cached log returns Parquet dataset."""
    path = Path(artifacts_dir) / "returns.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Cached returns file not found at {path}. Run data ingestion pipeline first."
        )
    return pd.read_parquet(path)


def load_cached_ohlcv(artifacts_dir: Path | str = "artifacts/data") -> pd.DataFrame:
    """Load cached OHLCV Parquet dataset."""
    path = Path(artifacts_dir) / "ohlcv.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Cached OHLCV file not found at {path}. Run data ingestion pipeline first."
        )
    return pd.read_parquet(path)
