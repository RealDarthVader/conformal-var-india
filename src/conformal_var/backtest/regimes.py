"""Indian market crisis regimes evaluation and stress test slice scorecard generator.

Slices out-of-sample Value-at-Risk predictions across configured historical Indian regimes:
1. calm_bull: 2017-01-01 to 2018-01-31
2. ilfs_liquidity_crisis: 2018-09-01 to 2019-03-31
3. covid_crash: 2020-02-01 to 2020-06-30
4. rate_hike_tightening: 2022-01-01 to 2022-12-31
5. full_sample: Complete out-of-sample test window

Also conducts a single-stock tail stress test on YESBANK.NS during its 2018-2020 collapse.
Produces complete scorecards (CSV and Markdown) with Kupiec, Christoffersen, and Basel metrics.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from tabulate import tabulate

from conformal_var.backtest.metrics import run_full_backtest
from conformal_var.config import AppConfig, load_config
from conformal_var.conformal.calibrator import AdaptiveConformalVaR, SlidingWindowConformalVaR
from conformal_var.conformal.vol_scaled import VolScaledConformalVaR
from conformal_var.data.loader import load_cached_ohlcv, load_cached_returns
from conformal_var.features.engineer import FeatureEngineer
from conformal_var.models.baselines import (
    GARCHNormalVaR,
    GARCHStudentTVaR,
    HistoricalSimulationVaR,
    RiskMetricsEWMAVaR,
)
from conformal_var.models.lgbm_quantile import LightGBMQuantileVaR

logger = logging.getLogger("regimes_backtest")


def compute_pinball_loss(
    returns: npt.NDArray[np.float64],
    var_preds: npt.NDArray[np.float64],
    alpha: float,
) -> float:
    """Compute mean quantile pinball loss."""
    diff = returns - var_preds
    violation = (returns < var_preds).astype(float)
    return float(np.mean((alpha - violation) * diff))


def run_models_for_series(
    ticker: str,
    returns_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    alphas: Sequence[float] = (0.05, 0.01),
    training_window: int = 756,
    calib_buffer: int = 252,
    refit_every: int = 63,
) -> tuple[pd.Series, dict[float, dict[str, pd.Series]]]:
    """Execute all 8 models out-of-sample and return date-indexed forecasts.

    Returns
    -------
    tuple[pd.Series, dict[float, dict[str, pd.Series]]]
        y_test: Realized out-of-sample returns series with DatetimeIndex.
        model_predictions: Mapping of alpha -> model_name -> VaR forecast series.
    """
    engineer = FeatureEngineer()
    X, y = engineer.create_dataset_from_cache(ticker, returns_df, ohlcv_df)

    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    n_samples = len(y_clean)
    if n_samples <= training_window:
        raise ValueError(f"Ticker {ticker} has {n_samples} samples, need > {training_window}")

    test_dates = y_clean.index[training_window:]
    y_test = y_clean.loc[test_dates]

    model_preds_by_alpha: dict[float, dict[str, pd.Series]] = {}

    for alpha in alphas:
        alpha_col = f"VaR_{int((1.0 - alpha) * 100)}"
        model_dict: dict[str, pd.Series] = {}

        # 1. Historical Simulation
        hs = HistoricalSimulationVaR(window=252)
        hs_df = hs.predict_rolling(y_clean, alphas=[alpha], min_history=training_window)
        model_dict["Historical Simulation (252d)"] = hs_df.loc[test_dates, alpha_col]

        # 2. RiskMetrics EWMA
        ewma = RiskMetricsEWMAVaR(decay_factor=0.94)
        ewma_df = ewma.predict_rolling(y_clean, alphas=[alpha], min_history=training_window)
        model_dict["RiskMetrics EWMA (0.94)"] = ewma_df.loc[test_dates, alpha_col]

        # 3. GARCH(1,1)-Normal
        gnorm = GARCHNormalVaR(training_window=training_window, refit_every=refit_every)
        gnorm_df = gnorm.predict_rolling(y_clean, alphas=[alpha], min_history=training_window)
        model_dict["GARCH(1,1)-Normal"] = gnorm_df.loc[test_dates, alpha_col]

        # 4. GARCH(1,1)-Student-t
        gt = GARCHStudentTVaR(
            training_window=training_window,
            refit_every=refit_every,
            dist="t",
        )
        gt_df = gt.predict_rolling(y_clean, alphas=[alpha], min_history=training_window)
        model_dict["GARCH(1,1)-Student-t"] = gt_df.loc[test_dates, alpha_col]

        # 5. Uncalibrated LightGBM
        lgbm = LightGBMQuantileVaR(
            alphas=[alpha],
            training_window=training_window,
            calibration_buffer=calib_buffer,
            refit_every=refit_every,
            max_depth=4,
            num_leaves=15,
            min_child_samples=30,
            learning_rate=0.03,
            n_estimators=100,
        )
        lgb_roll = lgbm.predict_rolling(X_clean, y_clean, alphas=[alpha])
        raw_lgb = lgb_roll.loc[test_dates, alpha_col]
        model_dict["Uncalibrated LightGBM"] = raw_lgb

        # 6. LightGBM + Sliding CQR
        cqr = SlidingWindowConformalVaR(alpha=alpha, window=calib_buffer)
        cqr_df = cqr.predict_sequence(raw_lgb, y_test)
        model_dict["LightGBM + Sliding CQR"] = pd.Series(
            cqr_df["conformal_var"].to_numpy(),
            index=test_dates,
        )

        # 7. LightGBM + Online ACI
        aci = AdaptiveConformalVaR(alpha=alpha, gamma=0.01, window=calib_buffer)
        aci_df = aci.predict_sequence(raw_lgb, y_test)
        model_dict["LightGBM + Online ACI"] = pd.Series(
            aci_df["conformal_var"].to_numpy(),
            index=test_dates,
        )

        # 8. LightGBM + Vol-Scaled ACI
        sigmas = X_clean.loc[test_dates, "realized_vol_21d"]
        vol_aci = VolScaledConformalVaR(
            alpha=alpha,
            gamma=0.01,
            window=calib_buffer,
        )
        vol_df = vol_aci.predict_sequence(raw_lgb, y_test, sigmas)
        model_dict["LightGBM + Vol-Scaled ACI"] = pd.Series(
            vol_df["conformal_var"].to_numpy(),
            index=test_dates,
        )

        model_preds_by_alpha[alpha] = model_dict

    return y_test, model_preds_by_alpha


def slice_scorecard(
    y_test: pd.Series,
    model_predictions: dict[float, dict[str, pd.Series]],
    regime_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    ticker: str = "^NSEI",
) -> list[dict[str, Any]]:
    """Compute performance scorecard metrics for a specific time slice/regime."""
    # Slice realized returns
    if start_date and end_date:
        y_slice = y_test.loc[start_date:end_date]
    elif start_date:
        y_slice = y_test.loc[start_date:]
    elif end_date:
        y_slice = y_test.loc[:end_date]
    else:
        y_slice = y_test

    if len(y_slice) == 0:
        logger.warning(
            "Regime slice %s (%s to %s) yielded 0 trading days for %s",
            regime_name,
            start_date,
            end_date,
            ticker,
        )
        return []

    r_arr = y_slice.to_numpy(dtype=np.float64)
    slice_records: list[dict[str, Any]] = []

    for alpha, models in model_predictions.items():
        for model_name, preds_series in models.items():
            var_slice = preds_series.loc[y_slice.index].to_numpy(dtype=np.float64)

            # Violations: return < VaR (where VaR is negative)
            violations = (r_arr < var_slice).astype(int)
            total_days = len(r_arr)
            n_violations = int(np.sum(violations))
            emp_coverage = 1.0 - (n_violations / total_days)
            sharpness = float(np.mean(np.abs(var_slice)))
            pinball = compute_pinball_loss(r_arr, var_slice, alpha)

            # Statistical risk-desk backtests
            report = run_full_backtest(violations, alpha=alpha, window_size=250)

            slice_records.append(
                {
                    "Ticker": ticker,
                    "Regime": regime_name,
                    "Start Date": str(y_slice.index[0])[:10],
                    "End Date": str(y_slice.index[-1])[:10],
                    "Days": total_days,
                    "Alpha": alpha,
                    "Model": model_name,
                    "Coverage": f"{emp_coverage * 100:.2f}%",
                    "Violations": n_violations,
                    "Sharpness": f"{sharpness:.4f}",
                    "Pinball Loss": f"{pinball:.6f}",
                    "Kupiec p-val": round(report.kupiec.p_value, 4),
                    "Kupiec Pass": report.kupiec.passed,
                    "Christoffersen p-val": round(report.independence.p_value, 4),
                    "Christoffersen Pass": report.independence.passed,
                    "Basel Zone": report.traffic_light.zone,
                    "Basel Mult": report.traffic_light.multiplier,
                    "_raw_coverage_pct": emp_coverage * 100,
                    "_raw_sharpness": sharpness,
                    "_raw_pinball": pinball,
                }
            )

    return slice_records


def generate_regime_scorecard(
    config: AppConfig | None = None,
    output_dir: str | Path = "artifacts",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete multi-regime scorecard on ^NSEI and YESBANK.NS stress test.

    Saves:
    1. artifacts/tables/regime_scorecard.csv
    2. artifacts/tables/regime_scorecard.md
    3. artifacts/tables/yesbank_stress_test.csv
    4. artifacts/tables/yesbank_stress_test.md
    """
    if config is None:
        config = load_config()

    out_path = Path(output_dir)
    tables_dir = out_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    returns_df = load_cached_returns(out_path / "data")
    ohlcv_df = load_cached_ohlcv(out_path / "data")

    # -----------------------------------------------------------------
    # Part 1: Index-level Market Regimes on ^NSEI
    # -----------------------------------------------------------------
    logger.info("Running out-of-sample models on ^NSEI across market regimes...")
    y_test_nsei, model_preds_nsei = run_models_for_series(
        "^NSEI",
        returns_df,
        ohlcv_df,
        alphas=config.target_quantiles,
        training_window=config.windows.training_window,
        calib_buffer=config.windows.calibration_window,
    )

    all_scorecards: list[dict[str, Any]] = []

    # 1. Full out-of-sample period
    all_scorecards.extend(
        slice_scorecard(
            y_test_nsei,
            model_preds_nsei,
            regime_name="full_sample_10y",
            ticker="^NSEI",
        )
    )

    # 2. Configured crisis regimes
    for reg_key, reg_val in config.market_regimes.items():
        all_scorecards.extend(
            slice_scorecard(
                y_test_nsei,
                model_preds_nsei,
                regime_name=reg_key,
                start_date=reg_val.start_date.isoformat(),
                end_date=reg_val.end_date.isoformat(),
                ticker="^NSEI",
            )
        )

    scorecard_df = pd.DataFrame(all_scorecards)

    # -----------------------------------------------------------------
    # Part 2: Single-Stock Tail Stress Test: YESBANK.NS (2018–2020)
    # -----------------------------------------------------------------
    logger.info("Running single-stock tail stress test on YESBANK.NS...")
    y_test_yes, model_preds_yes = run_models_for_series(
        "YESBANK.NS",
        returns_df,
        ohlcv_df,
        alphas=config.target_quantiles,
        training_window=config.windows.training_window,
        calib_buffer=config.windows.calibration_window,
    )

    yesbank_records = slice_scorecard(
        y_test_yes,
        model_preds_yes,
        regime_name="yesbank_collapse_2018_2020",
        start_date="2018-01-01",
        end_date="2020-12-31",
        ticker="YESBANK.NS",
    )
    yesbank_df = pd.DataFrame(yesbank_records)

    # -----------------------------------------------------------------
    # Save CSV and Markdown artifacts
    # -----------------------------------------------------------------
    display_cols = [
        "Ticker",
        "Regime",
        "Alpha",
        "Model",
        "Coverage",
        "Violations",
        "Sharpness",
        "Pinball Loss",
        "Kupiec p-val",
        "Christoffersen p-val",
        "Basel Zone",
    ]

    # Save Regime Scorecard
    scorecard_df.to_csv(tables_dir / "regime_scorecard.csv", index=False)
    filtered_scorecard = scorecard_df[display_cols]
    scorecard_md = tabulate(filtered_scorecard, headers="keys", tablefmt="github", showindex=False)
    with open(tables_dir / "regime_scorecard.md", "w", encoding="utf-8") as f:
        f.write("# Market Regimes VaR Backtest Scorecard (^NSEI)\n\n")
        f.write(scorecard_md)
        f.write("\n")

    # Save YES Bank Stress Test
    yesbank_df.to_csv(tables_dir / "yesbank_stress_test.csv", index=False)
    filtered_yesbank = yesbank_df[display_cols]
    yesbank_md = tabulate(filtered_yesbank, headers="keys", tablefmt="github", showindex=False)
    with open(tables_dir / "yesbank_stress_test.md", "w", encoding="utf-8") as f:
        f.write("# YESBANK.NS Left-Tail Collapse Stress Test (2018–2020)\n\n")
        f.write(yesbank_md)
        f.write("\n")

    logger.info("Saved regime scorecards and stress tests to %s", tables_dir)
    return scorecard_df, yesbank_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_regime_scorecard()
