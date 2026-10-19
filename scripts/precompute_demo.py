"""Offline precomputation script for the Streamlit demo.

Trains all 8 models on ^NSEI and YESBANK.NS, saves per-day VaR predictions,
realized returns, violation flags, and the ACI alpha_t path as Parquet files:

    artifacts/demo/nsei_predictions_alpha005.parquet
    artifacts/demo/nsei_predictions_alpha001.parquet
    artifacts/demo/yesbank_predictions_alpha005.parquet
    artifacts/demo/yesbank_predictions_alpha001.parquet
    artifacts/demo/portfolio_universe_returns.parquet   (13 equity tickers, no indices)

Columns per prediction file:
    date, realized_return,
    var_hist_sim, var_ewma, var_garch_n, var_garch_t,
    var_lgbm_raw, var_lgbm_cqr, var_lgbm_aci, var_lgbm_vol_aci,
    aci_alpha_t, violation_hist_sim, violation_ewma, violation_garch_n,
    violation_garch_t, violation_lgbm_raw, violation_lgbm_cqr,
    violation_lgbm_aci, violation_lgbm_vol_aci

Run once before launching the demo:
    python scripts/precompute_demo.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute_demo")

TRAINING_WINDOW = 756
CALIB_BUFFER = 252
REFIT_EVERY = 63
OUT_DIR = Path("artifacts/demo")


def run_for_ticker(
    ticker: str,
    returns_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Generate full walk-forward VaR prediction table for one ticker and alpha."""
    logger.info("  Processing %s @ alpha=%.2f", ticker, alpha)

    engineer = FeatureEngineer()
    X, y = engineer.create_dataset_from_cache(ticker, returns_df, ohlcv_df)

    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    n_samples = len(y_clean)
    if n_samples <= TRAINING_WINDOW:
        raise ValueError(f"{ticker}: only {n_samples} samples, need > {TRAINING_WINDOW}")

    test_dates = y_clean.index[TRAINING_WINDOW:]
    y_test = y_clean.loc[test_dates]
    alpha_col = f"VaR_{int((1.0 - alpha) * 100)}"

    # ── 1. Historical Simulation ──────────────────────────────────────────────
    hs = HistoricalSimulationVaR(window=252)
    hs_preds = hs.predict_rolling(y_clean, alphas=[alpha], min_history=TRAINING_WINDOW).loc[
        test_dates, alpha_col
    ]

    # ── 2. RiskMetrics EWMA ───────────────────────────────────────────────────
    ewma = RiskMetricsEWMAVaR(decay_factor=0.94)
    ewma_preds = ewma.predict_rolling(y_clean, alphas=[alpha], min_history=TRAINING_WINDOW).loc[
        test_dates, alpha_col
    ]

    # ── 3. GARCH(1,1)-Normal ──────────────────────────────────────────────────
    gn = GARCHNormalVaR(training_window=TRAINING_WINDOW, refit_every=REFIT_EVERY)
    gn_preds = gn.predict_rolling(y_clean, alphas=[alpha], min_history=TRAINING_WINDOW).loc[
        test_dates, alpha_col
    ]

    # ── 4. GARCH(1,1)-Student-t ───────────────────────────────────────────────
    gt = GARCHStudentTVaR(training_window=TRAINING_WINDOW, refit_every=REFIT_EVERY, dist="t")
    gt_preds = gt.predict_rolling(y_clean, alphas=[alpha], min_history=TRAINING_WINDOW).loc[
        test_dates, alpha_col
    ]

    # ── 5. Uncalibrated LightGBM ──────────────────────────────────────────────
    lgbm = LightGBMQuantileVaR(
        alphas=[alpha],
        training_window=TRAINING_WINDOW,
        calibration_buffer=CALIB_BUFFER,
        refit_every=REFIT_EVERY,
        max_depth=4,
        num_leaves=15,
        min_child_samples=30,
        learning_rate=0.03,
        n_estimators=100,
    )
    lgb_roll = lgbm.predict_rolling(X_clean, y_clean, alphas=[alpha])
    raw_lgb_preds = lgb_roll.loc[test_dates, alpha_col]

    # ── 6. LightGBM + Sliding CQR ────────────────────────────────────────────
    cqr = SlidingWindowConformalVaR(alpha=alpha, window=CALIB_BUFFER)
    cqr_df = cqr.predict_sequence(raw_lgb_preds, y_test)
    cqr_preds = cqr_df["conformal_var"]

    # ── 7. LightGBM + Online ACI ─────────────────────────────────────────────
    aci = AdaptiveConformalVaR(alpha=alpha, gamma=0.01, window=CALIB_BUFFER)
    aci_df = aci.predict_sequence(raw_lgb_preds, y_test)
    aci_preds = aci_df["conformal_var"]
    aci_alpha_t = aci_df["alpha_t"]  # the dynamic miscoverage level

    # ── 8. LightGBM + Vol-Scaled ACI ─────────────────────────────────────────
    vol_col = "realized_vol_21d"
    sigmas_test = X_clean.loc[test_dates, vol_col]
    vol_aci = VolScaledConformalVaR(alpha=alpha, gamma=0.01, window=CALIB_BUFFER, adaptive=True)
    vol_aci_df = vol_aci.predict_sequence(raw_lgb_preds, y_test, sigmas_test)
    vol_aci_preds = vol_aci_df["conformal_var"]

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    out = pd.DataFrame(
        {
            "realized_return": y_test,
            "var_hist_sim": hs_preds,
            "var_ewma": ewma_preds,
            "var_garch_n": gn_preds,
            "var_garch_t": gt_preds,
            "var_lgbm_raw": raw_lgb_preds,
            "var_lgbm_cqr": cqr_preds,
            "var_lgbm_aci": aci_preds,
            "var_lgbm_vol_aci": vol_aci_preds,
            "aci_alpha_t": aci_alpha_t,
        },
        index=test_dates,
    )
    out.index.name = "date"

    # Violation flags (1 = loss exceeded VaR, i.e. bad day)
    var_cols = [c for c in out.columns if c.startswith("var_")]
    for vc in var_cols:
        flag_col = vc.replace("var_", "violation_")
        out[flag_col] = (out["realized_return"] < out[vc]).astype(np.int8)

    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading cached returns and OHLCV data...")
    returns_df = load_cached_returns()
    ohlcv_df = load_cached_ohlcv()

    tickers = {
        "nsei": "^NSEI",
        "yesbank": "YESBANK.NS",
    }

    for name, ticker in tickers.items():
        for alpha in (0.05, 0.01):
            tag = f"alpha{int(alpha * 100):03d}"
            out_path = OUT_DIR / f"{name}_predictions_{tag}.parquet"
            if out_path.exists():
                logger.info("  Skipping %s (already exists) — delete to recompute.", out_path)
                continue
            try:
                df = run_for_ticker(ticker, returns_df, ohlcv_df, alpha)
                df.to_parquet(out_path, index=True)
                logger.info("  Saved %s  (%d rows, %d cols)", out_path, *df.shape)
            except Exception as exc:
                logger.exception("  FAILED %s @ %.2f: %s", ticker, alpha, exc)

    # ── Portfolio universe: 13 equity tickers (no index, no VIX) ─────────────
    equity_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "KOTAKBANK.NS",
        "YESBANK.NS", "IDEA.NS", "ZEEL.NS",
    ]
    avail = [t for t in equity_tickers if t in returns_df.columns]
    port_path = OUT_DIR / "portfolio_universe_returns.parquet"
    returns_df[avail].to_parquet(port_path, index=True)
    logger.info("Saved portfolio universe returns → %s  (%d tickers)", port_path, len(avail))

    logger.info("Precomputation complete. Artifacts in %s/", OUT_DIR)


if __name__ == "__main__":
    main()
