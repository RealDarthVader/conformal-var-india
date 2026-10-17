"""Tests for Indian equity data ingestion, calendar alignment, and leakage prevention."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conformal_var.data.loader import (
    CORE_LIQUID_CONSTITUENTS,
    MARKET_BENCHMARK,
    NSEDataLoader,
    load_cached_ohlcv,
    load_cached_returns,
)


@pytest.fixture(scope="module")
def cached_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixture providing cached returns and OHLCV datasets."""
    artifacts_dir = Path("artifacts/data")
    returns_exist = (artifacts_dir / "returns.parquet").exists()
    ohlcv_exist = (artifacts_dir / "ohlcv.parquet").exists()
    if not returns_exist or not ohlcv_exist:
        loader = NSEDataLoader()
        return loader.run_pipeline(artifacts_dir=artifacts_dir)
    return load_cached_returns(artifacts_dir), load_cached_ohlcv(artifacts_dir)


def test_exact_index_alignment_with_nsei(
    cached_datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Verify that all universe series are strictly aligned to ^NSEI trading days."""
    returns_df, ohlcv_df = cached_datasets

    # Check identical temporal index across returns and OHLCV
    assert (returns_df.index == ohlcv_df.index).all(), "Returns and OHLCV indices must be identical"

    # Must be monotonically increasing datetime index
    assert returns_df.index.is_monotonic_increasing, (
        "Index must be strictly monotonically increasing"
    )
    assert not returns_df.index.has_duplicates, "Index must not contain duplicate trading days"

    # Benchmark must be present
    assert MARKET_BENCHMARK in returns_df.columns

    # Verify span covers 15+ years (2010 to 2026)
    assert returns_df.index[0].date() <= date(2010, 1, 15)
    assert returns_df.index[-1].date() >= date(2026, 1, 1)
    assert len(returns_df) > 3500, f"Expected 3500+ NSE trading days, got {len(returns_df)}"


def test_zero_nans_in_active_windows(cached_datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verify zero NaN values in active trading windows for core liquid constituents."""
    returns_df, _ = cached_datasets

    # For benchmark and core liquid tickers, active window starts from index 1 (day 0 is diff NaN)
    for ticker in [MARKET_BENCHMARK, *CORE_LIQUID_CONSTITUENTS]:
        assert ticker in returns_df.columns, f"Expected {ticker} in universe"
        series = returns_df[ticker]

        # First valid index after ticker inception
        first_valid = series.first_valid_index()
        assert first_valid is not None, f"Ticker {ticker} has no valid returns"

        # From first valid index to latest, core liquid stocks must have 0 NaNs
        active_window = series.loc[first_valid:]
        nan_count = active_window.isna().sum()
        assert nan_count == 0, f"Found {nan_count} unexpected NaNs in active window for {ticker}"


def test_absence_of_future_data_leakage(cached_datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Verify that returns calculation does not look ahead or leak future data."""
    returns_df, ohlcv_df = cached_datasets

    # 1. Day 0 return must be NaN because t-1 does not exist in dataset
    first_date = returns_df.index[0]
    for ticker in [MARKET_BENCHMARK, *CORE_LIQUID_CONSTITUENTS]:
        assert np.isnan(returns_df.loc[first_date, ticker]), "First return observation must be NaN"

    # 2. Strict causal calculation check:
    # r_t = ln(P_t / P_{t-1}). Check random sample dates
    close_prices = ohlcv_df["Adj Close"] if "Adj Close" in ohlcv_df.columns else ohlcv_df["Close"]

    for i in [1, 50, 500, 1500, 3000]:
        t_date = returns_df.index[i]
        t_prev = returns_df.index[i - 1]

        p_t = close_prices.loc[t_date, MARKET_BENCHMARK]
        p_prev = close_prices.loc[t_prev, MARKET_BENCHMARK]

        expected_return = np.log(p_t / p_prev)
        actual_return = returns_df.loc[t_date, MARKET_BENCHMARK]

        assert np.isclose(expected_return, actual_return, atol=1e-6), (
            f"Leakage or calculation mismatch at {t_date}: "
            f"expected {expected_return}, got {actual_return}"
        )


def test_forward_fill_limit_rule() -> None:
    """Verify forward-fill maximum limit of 3 days and refusal to bridge prolonged suspensions."""
    # Create synthetic daily prices aligned to a benchmark
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    benchmark = pd.Series([100.0 + i for i in range(10)], index=dates)

    # Stock A has a 2-day gap (days 2 and 3 missing) -> should be filled
    stock_a = pd.Series(
        [50.0, 51.0, np.nan, np.nan, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0],
        index=dates,
    )

    # Stock B has a 4-day gap (days 2, 3, 4, 5 missing) -> day 5 should remain NaN (limit=3)
    stock_b = pd.Series(
        [50.0, 51.0, np.nan, np.nan, np.nan, np.nan, 56.0, 57.0, 58.0, 59.0],
        index=dates,
    )

    raw_data = {
        ("Close", "^NSEI"): benchmark,
        ("Close", "STOCK_A"): stock_a,
        ("Close", "STOCK_B"): stock_b,
    }
    raw_df = pd.DataFrame(raw_data, index=dates)
    raw_df.columns = pd.MultiIndex.from_tuples(raw_df.columns)

    loader = NSEDataLoader(tickers=["^NSEI", "STOCK_A", "STOCK_B"], max_ffill_days=3)
    cleaned = loader.clean_and_align_ohlcv(raw_df)

    # Stock A should have no NaNs after day 0
    assert not cleaned["Close"]["STOCK_A"].isna().any(), "2-day gap should be completely filled"

    # Stock B should have exactly 1 NaN remaining at day 5 (since limit=3 allows days 2, 3, 4 only)
    stock_b_cleaned = cleaned["Close"]["STOCK_B"]
    msg = "4th consecutive missing day must NOT be forward-filled"
    assert np.isnan(stock_b_cleaned.iloc[5]), msg


def test_corporate_action_anomaly_detection() -> None:
    """Verify anomaly detection flags unadjusted split jumps > 40% without intraday confirmation."""
    dates = pd.date_range("2024-01-01", periods=4, freq="B")

    # Simulate an unadjusted 2-for-1 split: Close drops 50% (unadj return ~ -0.693),
    # while Adj Close remains continuous (adj return = 0.0)
    data = {
        ("Close", "SPLIT_CO"): [100.0, 100.0, 50.0, 51.0],
        ("Adj Close", "SPLIT_CO"): [50.0, 50.0, 50.0, 51.0],
        ("Volume", "SPLIT_CO"): [1000.0, 1000.0, 1000.0, 1000.0],
        ("High", "SPLIT_CO"): [101.0, 101.0, 52.0, 52.0],
        ("Low", "SPLIT_CO"): [99.0, 99.0, 49.0, 50.0],
    }
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    loader = NSEDataLoader(tickers=["SPLIT_CO"], anomaly_threshold=0.40)
    anomalies = loader.detect_corporate_action_anomalies(df)

    assert len(anomalies) == 1
    report = anomalies[0]
    assert report.ticker == "SPLIT_CO"
    assert "Unadjusted split/bonus jump" in report.flag_reason
    assert report.unadjusted_return < -0.60
