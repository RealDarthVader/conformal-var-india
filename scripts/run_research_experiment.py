"""Research experiment script evaluating ACI against GARCH and Vol-Scaled residual bands.

Core Research Hypothesis:
'Does Adaptive Conformal Inference (ACI) over LightGBM actually beat a well-tuned
GARCH(1,1)-t or volatility-scaled residual band?'

Ablation benchmark comparing 8 models on Indian equities at alpha in {0.05, 0.01}:
1. Historical Simulation (252d)
2. RiskMetrics EWMA
3. GARCH(1,1)-Normal
4. GARCH(1,1)-Student-t
5. Uncalibrated LightGBM Quantile
6. LightGBM + Sliding CQR
7. LightGBM + Online ACI (gamma=0.01)
8. LightGBM + Vol-Scaled ACI (gamma=0.01)

Metrics Tracked:
- Empirical Coverage: 1 - alpha_hat
- Coverage Error: |alpha_hat - alpha|
- Sharpness: Mean |VaR_t| (smaller at valid coverage is sharper/better)
- Quantile Pinball Loss: L_alpha(r_t, v_t) = (alpha - I{r_t < v_t})(r_t - v_t)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conformal_var.conformal.calibrator import AdaptiveConformalVaR, SlidingWindowConformalVaR
from conformal_var.conformal.vol_scaled import VolScaledConformalVaR
from conformal_var.data.loader import (
    DEFAULT_UNIVERSE,
    load_cached_ohlcv,
    load_cached_returns,
)
from conformal_var.features.engineer import FeatureEngineer
from conformal_var.models.baselines import (
    GARCHNormalVaR,
    GARCHStudentTVaR,
    HistoricalSimulationVaR,
    RiskMetricsEWMAVaR,
)
from conformal_var.models.lgbm_quantile import LightGBMQuantileVaR

logger = logging.getLogger("research_experiment")


def compute_pinball_loss(r: np.ndarray, var_pred: np.ndarray, alpha: float) -> float:
    """Compute mean quantile pinball loss.

    L_alpha(r_t, v_t) = (alpha - I{r_t < v_t}) * (r_t - v_t)
    """
    diff = r - var_pred
    violation = (r < var_pred).astype(float)
    loss = (alpha - violation) * diff
    return float(np.mean(loss))


def evaluate_predictions(
    realized_returns: np.ndarray,
    var_predictions: np.ndarray,
    alpha: float,
    model_name: str,
) -> dict[str, Any]:
    """Calculate core performance metrics for a model's VaR forecasts."""
    violations = realized_returns < var_predictions
    emp_alpha = float(np.mean(violations))
    emp_coverage = 1.0 - emp_alpha
    cov_error = abs(emp_alpha - alpha)
    sharpness = float(np.mean(np.abs(var_predictions)))
    pinball = compute_pinball_loss(realized_returns, var_predictions, alpha)

    return {
        "Model": model_name,
        "Target Alpha": alpha,
        "Empirical Coverage": f"{emp_coverage * 100:.2f}%",
        "Coverage Error": f"{cov_error * 100:.2f}%",
        "Sharpness": f"{sharpness:.4f}",
        "Pinball Loss": f"{pinball:.6f}",
        "_raw_cov_error": cov_error,
        "_raw_pinball": pinball,
        "_raw_sharpness": sharpness,
    }


def run_experiment_for_asset(
    ticker: str,
    returns_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    alphas: tuple[float, ...] = (0.05, 0.01),
    training_window: int = 756,
    calib_buffer: int = 252,
    refit_every: int = 63,
) -> pd.DataFrame:
    """Run all 8 models through walk-forward evaluation on a single asset."""
    logger.info("Running research experiment for ticker: %s", ticker)

    engineer = FeatureEngineer()
    X, y = engineer.create_dataset_from_cache(ticker, returns_df, ohlcv_df)

    # Valid data index
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    n_samples = len(y_clean)
    if n_samples <= training_window:
        raise ValueError(
            f"Asset {ticker} has only {n_samples} samples, need > {training_window}"
        )

    # Common evaluation test dates
    test_dates = y_clean.index[training_window:]
    y_test = y_clean.loc[test_dates].to_numpy(dtype=np.float64)

    all_results: list[dict[str, Any]] = []

    for alpha in alphas:
        alpha_col = f"VaR_{int((1.0 - alpha) * 100)}"
        logger.info(
            "Evaluating models for alpha = %.2f (confidence = %.0f%%)...",
            alpha,
            (1 - alpha) * 100,
        )

        # -------------------------------------------------------------
        # 1. Historical Simulation (252d)
        # -------------------------------------------------------------
        hs_model = HistoricalSimulationVaR(window=252)
        hs_preds = hs_model.predict_rolling(
            y_clean,
            alphas=[alpha],
            min_history=training_window,
        ).loc[test_dates, alpha_col].to_numpy()
        all_results.append(
            evaluate_predictions(y_test, hs_preds, alpha, "1. Historical Simulation (252d)")
        )

        # -------------------------------------------------------------
        # 2. RiskMetrics EWMA
        # -------------------------------------------------------------
        ewma_model = RiskMetricsEWMAVaR(decay_factor=0.94)
        ewma_preds = ewma_model.predict_rolling(
            y_clean,
            alphas=[alpha],
            min_history=training_window,
        ).loc[test_dates, alpha_col].to_numpy()
        all_results.append(
            evaluate_predictions(y_test, ewma_preds, alpha, "2. RiskMetrics EWMA (0.94)")
        )

        # -------------------------------------------------------------
        # 3. GARCH(1,1)-Normal
        # -------------------------------------------------------------
        garch_norm = GARCHNormalVaR(training_window=training_window, refit_every=refit_every)
        gnorm_preds = garch_norm.predict_rolling(
            y_clean,
            alphas=[alpha],
            min_history=training_window,
        ).loc[test_dates, alpha_col].to_numpy()
        all_results.append(
            evaluate_predictions(y_test, gnorm_preds, alpha, "3. GARCH(1,1)-Normal")
        )

        # -------------------------------------------------------------
        # 4. GARCH(1,1)-Student-t
        # -------------------------------------------------------------
        garch_t = GARCHStudentTVaR(
            training_window=training_window,
            refit_every=refit_every,
            dist="t",
        )
        gt_preds = garch_t.predict_rolling(
            y_clean,
            alphas=[alpha],
            min_history=training_window,
        ).loc[test_dates, alpha_col].to_numpy()
        all_results.append(
            evaluate_predictions(y_test, gt_preds, alpha, "4. GARCH(1,1)-Student-t")
        )

        # -------------------------------------------------------------
        # 5. Uncalibrated LightGBM Quantile
        # -------------------------------------------------------------
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
        lgb_roll_df = lgbm.predict_rolling(X_clean, y_clean, alphas=[alpha])
        raw_lgb_preds = lgb_roll_df.loc[test_dates, alpha_col]
        all_results.append(
            evaluate_predictions(
                y_test, raw_lgb_preds.to_numpy(), alpha, "5. Uncalibrated LightGBM"
            )
        )

        # -------------------------------------------------------------
        # 6. LightGBM + Sliding CQR (window=252)
        # -------------------------------------------------------------
        cqr = SlidingWindowConformalVaR(alpha=alpha, window=calib_buffer)
        y_test_ser = y_clean.loc[test_dates]
        cqr_df = cqr.predict_sequence(raw_lgb_preds, y_test_ser)
        all_results.append(
            evaluate_predictions(
                y_test,
                cqr_df["conformal_var"].to_numpy(),
                alpha,
                "6. LightGBM + Sliding CQR",
            )
        )

        # -------------------------------------------------------------
        # 7. LightGBM + Online ACI (gamma=0.01)
        # -------------------------------------------------------------
        aci = AdaptiveConformalVaR(alpha=alpha, gamma=0.01, window=calib_buffer)
        aci_df = aci.predict_sequence(raw_lgb_preds, y_test_ser)
        all_results.append(
            evaluate_predictions(
                y_test,
                aci_df["conformal_var"].to_numpy(),
                alpha,
                "7. LightGBM + Online ACI",
            )
        )

        # -------------------------------------------------------------
        # 8. LightGBM + Vol-Scaled ACI (gamma=0.01)
        # -------------------------------------------------------------
        vol_col = "realized_vol_21d"
        sigmas_test = X_clean.loc[test_dates, vol_col]
        vol_aci = VolScaledConformalVaR(
            alpha=alpha,
            gamma=0.01,
            window=calib_buffer,
            adaptive=True,
        )
        vol_aci_df = vol_aci.predict_sequence(raw_lgb_preds, y_test_ser, sigmas_test)
        all_results.append(
            evaluate_predictions(
                y_test,
                vol_aci_df["conformal_var"].to_numpy(),
                alpha,
                "8. LightGBM + Vol-Scaled ACI",
            )
        )

    res_df = pd.DataFrame(all_results)
    return res_df


def main() -> int:
    """Execute research experiment and print console comparison table."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Core Research Experiment: ACI vs GARCH vs Vol-Scaled Conformal VaR."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="^NSEI",
        help="Asset ticker to evaluate (default: ^NSEI)",
    )
    parser.add_argument(
        "--all-assets",
        action="store_true",
        help="Run experiment across all liquid and stressed constituents.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/tables",
        help="Directory to persist markdown results table.",
    )
    args = parser.parse_args()

    returns_df = load_cached_returns()
    ohlcv_df = load_cached_ohlcv()

    tickers_to_run = [args.ticker] if not args.all_assets else [
        t for t in DEFAULT_UNIVERSE if t != "^INDIAVIX" and t in returns_df.columns
    ]

    all_tables: list[pd.DataFrame] = []

    for tkr in tickers_to_run:
        try:
            tbl = run_experiment_for_asset(tkr, returns_df, ohlcv_df)
            tbl.insert(0, "Ticker", tkr)
            all_tables.append(tbl)
        except Exception as exc:
            logger.exception("Failed experiment for %s: %s", tkr, exc)

    if not all_tables:
        logger.error("No experiment results generated.")
        return 1

    combined_df = pd.concat(all_tables, ignore_index=True)
    display_cols = [
        "Ticker",
        "Model",
        "Target Alpha",
        "Empirical Coverage",
        "Coverage Error",
        "Sharpness",
        "Pinball Loss",
    ]
    summary_table = combined_df[display_cols]

    # Print formatted console table
    print("\n" + "=" * 95)
    print("CORE RESEARCH HYPOTHESIS COMPARISON: ACI vs GARCH vs VOL-SCALED CONFORMAL VAR")
    print("=" * 95)
    print(summary_table.to_string(index=False))
    print("=" * 95 + "\n")

    # Save to artifacts/tables
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "research_experiment_results.md"
    summary_table.to_markdown(out_file, index=False)
    logger.info("Saved benchmark report to %s", out_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
