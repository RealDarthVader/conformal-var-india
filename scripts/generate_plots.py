"""Publication-quality diagnostic plot generation script for conformal-var-india.

Generates three high-resolution diagnostic plots in artifacts/plots/:
1. march_2020_crash_backtest.png:
   Daily Nifty 50 returns from Jan 2020 to Jul 2020 overlaid with 99% VaR bands
   (Historical Simulation, GARCH(1,1)-t, Split Conformal, and Online ACI),
   marking violation days with red scatter points, plus dynamic alpha_t adaptation
   trajectory in the bottom subplot.
2. coverage_vs_sharpness_pareto.png:
   Scatter plot of Empirical Coverage Error (|alpha_hat - 0.01|) on the x-axis vs.
   Mean VaR Band Width (Sharpness) on the y-axis across all models.
3. basel_violations_by_regime.png:
   Grouped bar chart of violation rates across the 4 Indian market regimes with
   horizontal dashed lines marking nominal 1% target and Basel Yellow/Red thresholds.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure non-interactive backend
matplotlib.use("Agg")

from conformal_var.conformal.calibrator import (
    AdaptiveConformalVaR,
    SplitConformalVaR,
)
from conformal_var.data.loader import load_cached_ohlcv, load_cached_returns
from conformal_var.features.engineer import FeatureEngineer
from conformal_var.models.baselines import (
    GARCHStudentTVaR,
    HistoricalSimulationVaR,
)
from conformal_var.models.lgbm_quantile import LightGBMQuantileVaR

logger = logging.getLogger("generate_plots")


def set_custom_style() -> None:
    """Set publication grade styling for financial plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "figure.dpi": 300,
        }
    )


def generate_march_2020_crash_plot(
    returns_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate march_2020_crash_backtest.png."""
    logger.info("Generating plot 1: March 2020 Crash Backtest...")
    engineer = FeatureEngineer()
    X, y = engineer.create_dataset_from_cache("^NSEI", returns_df, ohlcv_df)

    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    training_window = 756
    alpha = 0.01

    # 1. Historical Simulation
    hs = HistoricalSimulationVaR(window=252)
    hs_preds = hs.predict_rolling(y_clean, alphas=[alpha], min_history=training_window)
    hs_series = hs_preds["VaR_99"]

    # 2. GARCH(1,1)-Student-t
    gt = GARCHStudentTVaR(training_window=training_window, dist="t")
    gt_preds = gt.predict_rolling(y_clean, alphas=[alpha], min_history=training_window)
    gt_series = gt_preds["VaR_99"]

    # 3. Base LightGBM Quantile
    lgbm = LightGBMQuantileVaR(
        alphas=[alpha],
        training_window=training_window,
        calibration_buffer=252,
        refit_every=63,
    )
    lgb_preds = lgbm.predict_rolling(X_clean, y_clean, alphas=[alpha])
    base_q_series = lgb_preds["VaR_99"]

    # 4. Split Conformal VaR
    # Fit split conformal using 2019 data (pre-crash calm period)
    base_calib = base_q_series.loc["2019-01-01":"2019-12-31"]
    y_calib = y_clean.loc["2019-01-01":"2019-12-31"]

    split_conf = SplitConformalVaR(alpha=alpha)
    split_conf.calibrate(base_calib, y_calib)
    split_preds = split_conf.predict(base_q_series)

    # 5. Online Adaptive Conformal VaR (ACI)
    aci = AdaptiveConformalVaR(alpha=alpha, gamma=0.01, window=252)
    aci.calibrate(base_calib, y_calib)

    # Filter strictly to Jan 2020 to Jul 2020
    y_plot = y_clean.loc["2020-01-01":"2020-07-31"]
    base_plot = base_q_series.loc["2020-01-01":"2020-07-31"]


    aci_df = aci.predict_sequence(base_plot, y_plot)
    aci_series = aci_df["conformal_var"]
    alpha_t_series = aci_df["alpha_t"]

    dates = y_plot.index
    r_vals = y_plot.to_numpy()

    # Create figure with 2 subplots (top: returns & VaR bands, bottom: alpha_t trajectory)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.8, 1.2]}
    )

    # Subplot 1: Realized Returns & VaR Models
    ax1.plot(dates, r_vals, color="#495057", lw=1.2, label="Nifty 50 Realized Log Return", zorder=3)
    ax1.plot(
        dates,
        hs_series.loc[dates],
        color="#fd7e14",
        ls="--",
        lw=1.6,
        label="Historical Simulation (252d)",
        alpha=0.9,
    )
    ax1.plot(
        dates,
        gt_series.loc[dates],
        color="#20c997",
        ls="-.",
        lw=1.6,
        label="GARCH(1,1)-Student-t",
        alpha=0.9,
    )
    ax1.plot(
        dates,
        split_preds.loc[dates],
        color="#6f42c1",
        ls=":",
        lw=1.8,
        label="Split Conformal (Pre-Crash Calib)",
        alpha=0.9,
    )
    ax1.plot(
        dates,
        aci_series,
        color="#0d6efd",
        ls="-",
        lw=2.2,
        label=r"Online ACI ($\gamma=0.01$)",
        zorder=4,
    )

    # Violations under Online ACI (r_t < VaR_t)
    aci_violations = r_vals < aci_series.to_numpy()
    if np.any(aci_violations):
        ax1.scatter(
            dates[aci_violations],
            r_vals[aci_violations],
            color="#dc3545",
            s=45,
            edgecolors="black",
            zorder=5,
            label="ACI VaR Violation",
        )

    ax1.set_title(
        r"COVID-19 Crash Backtest (Jan 2020 – Jul 2020): 99% 1-Day-Ahead VaR ($\alpha=0.01$)",
        fontweight="bold",
    )
    ax1.set_ylabel("Daily Return / VaR Forecast")
    ax1.axhline(0, color="gray", lw=0.6, ls="-")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.95)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Dynamic alpha_t trajectory
    ax2.plot(dates, alpha_t_series, color="#d63384", lw=1.8, label=r"Dynamic $\alpha_t$ Trajectory")
    ax2.axhline(
        alpha,
        color="black",
        ls="--",
        lw=1.0,
        label=r"Nominal $\alpha = 0.01$",
    )
    ax2.set_ylabel(r"Effective $\alpha_t$")
    ax2.set_xlabel("Trading Date")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "march_2020_crash_backtest.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved %s", output_path)


def generate_coverage_vs_sharpness_pareto_plot(
    output_dir: Path,
) -> None:
    """Generate coverage_vs_sharpness_pareto.png."""
    logger.info("Generating plot 2: Coverage vs Sharpness Pareto Frontier...")
    benchmark_csv = Path("artifacts/tables/regime_scorecard.csv")
    if not benchmark_csv.exists():
        raise FileNotFoundError(f"Missing {benchmark_csv}. Run regimes.py first.")

    df = pd.read_csv(benchmark_csv)
    # Filter to full_sample_10y and alpha=0.01
    full_df = df[(df["Regime"] == "full_sample_10y") & (df["Alpha"] == 0.01)].copy()

    # Extract Coverage Error: |empirical_coverage - 0.99| = |empirical_failure_rate - 0.01|
    full_df["Empirical_Rate"] = 1.0 - (full_df["Coverage"].str.rstrip("%").astype(float) / 100.0)
    full_df["Coverage_Error"] = (full_df["Empirical_Rate"] - 0.01).abs()
    full_df["Sharpness_Val"] = full_df["Sharpness"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    palette = sns.color_palette("tab10", len(full_df))
    markers = ["o", "s", "^", "D", "v", "<", "P", "X"]

    for idx, (_, row) in enumerate(full_df.iterrows()):
        m_name = row["Model"]
        x_val = row["Coverage_Error"] * 100  # Percentage error
        y_val = row["Sharpness_Val"]

        ax.scatter(
            x_val,
            y_val,
            color=palette[idx],
            marker=markers[idx % len(markers)],
            s=120,
            edgecolors="black",
            alpha=0.9,
            label=m_name,
            zorder=4,
        )
        # Label offset annotations
        offset_x = 0.02
        offset_y = 0.0003
        if "Vol-Scaled" in m_name:
            offset_y = -0.0007
        elif "Online ACI" in m_name:
            offset_x = -0.15
        ax.annotate(
            m_name,
            (x_val + offset_x, y_val + offset_y),
            fontsize=8.5,
            fontweight="semibold",
            color="#212529",
        )

    ax.set_title(
        r"Model Efficiency Pareto Frontier: 99% VaR ($\alpha = 0.01$) Over 10-Year OOS Nifty 50",
        fontweight="bold",
    )
    ax.set_xlabel(r"Empirical Coverage Error $|\hat{\alpha} - 0.01|$ (%) [Lower is Better]")
    ax.set_ylabel(r"Sharpness (Mean $|\widehat{\mathrm{VaR}}_t|$) [Lower is Better/Sharper]")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Shade ideal zone
    ax.axvspan(0.0, 0.25, color="#d1e7dd", alpha=0.3, label="Ideal Coverage Zone (< 0.25% Error)")

    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    output_path = output_dir / "coverage_vs_sharpness_pareto.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved %s", output_path)


def generate_basel_violations_by_regime_plot(
    output_dir: Path,
) -> None:
    """Generate basel_violations_by_regime.png."""
    logger.info("Generating plot 3: Basel Violations Across Regimes...")
    scorecard_csv = Path("artifacts/tables/regime_scorecard.csv")
    if not scorecard_csv.exists():
        raise FileNotFoundError(f"Missing {scorecard_csv}")

    df = pd.read_csv(scorecard_csv)
    # Select alpha=0.01 and 4 regimes
    regimes = ["calm_bull", "ilfs_liquidity_crisis", "covid_crash", "rate_hike_tightening"]
    regime_labels = {
        "calm_bull": "Calm Bull\n(2017-18)",
        "ilfs_liquidity_crisis": "IL&FS Crisis\n(2018-19)",
        "covid_crash": "COVID Crash\n(2020)",
        "rate_hike_tightening": "Rate Tightening\n(2022)",
    }

    sub_df = df[(df["Alpha"] == 0.01) & (df["Regime"].isin(regimes))].copy()
    sub_df["Failure_Rate_Pct"] = (
        1.0 - (sub_df["Coverage"].str.rstrip("%").astype(float) / 100.0)
    ) * 100.0
    sub_df["Regime_Label"] = sub_df["Regime"].map(regime_labels)

    # Selected representative models for clear visualization
    selected_models = [
        "Historical Simulation (252d)",
        "GARCH(1,1)-Student-t",
        "Uncalibrated LightGBM",
        "LightGBM + Online ACI",
        "LightGBM + Vol-Scaled ACI",
    ]
    sub_df = sub_df[sub_df["Model"].isin(selected_models)]

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.barplot(
        data=sub_df,
        x="Regime_Label",
        y="Failure_Rate_Pct",
        hue="Model",
        palette="crest",
        ax=ax,
        edgecolor="black",
        linewidth=0.7,
    )

    # Benchmark regulatory thresholds:
    # Target: 1.0% failure rate
    ax.axhline(1.0, color="#198754", ls="--", lw=1.8, label="Target 1.0% Rate (Nominal alpha)")
    # Basel Yellow Zone threshold: 5 violations in 250 days = 2.0%
    ax.axhline(2.0, color="#ffc107", ls="--", lw=1.8, label="Basel Yellow Zone Threshold (2.0%)")
    # Basel Red Zone threshold: 10 violations in 250 days = 4.0%
    ax.axhline(4.0, color="#dc3545", ls="--", lw=1.8, label="Basel Red Zone Threshold (4.0%)")

    ax.set_title(
        r"Empirical Failure Rates ($\alpha = 0.01$) Across Indian Market Crisis Regimes",
        fontweight="bold",
    )
    ax.set_xlabel("Market Regime")
    ax.set_ylabel("Empirical Failure Rate (%)")
    ax.legend(title="Risk Model", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = output_dir / "basel_violations_by_regime.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved %s", output_path)


def main() -> None:
    """Generate all publication plots."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_custom_style()

    out_dir = Path("artifacts/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    returns_df = load_cached_returns(Path("artifacts/data"))
    ohlcv_df = load_cached_ohlcv(Path("artifacts/data"))

    generate_march_2020_crash_plot(returns_df, ohlcv_df, out_dir)
    generate_coverage_vs_sharpness_pareto_plot(out_dir)
    generate_basel_violations_by_regime_plot(out_dir)
    logger.info("All diagnostic plots generated successfully in %s", out_dir)


if __name__ == "__main__":
    main()
