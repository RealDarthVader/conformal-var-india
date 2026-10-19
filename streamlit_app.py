"""Conformal VaR India — Interactive Streamlit Demo.

Tabs:
  1. 🔴 Crash Replay   — Returns + VaR bands + violations + ACI α_t path
  2. 📊 Model Comparison — Coverage vs Width scatter and full metrics table
  3. 💼 Portfolio Calculator — Weighted portfolio VaR / ES from pre-baked returns

All data is precomputed. No model fitting happens at request time.
Run `python scripts/precompute_demo.py` once to generate artifacts/demo/*.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── ensure src/ is importable when launched from repo root ───────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Conformal VaR India",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_DIR = Path("artifacts/demo")
TABLE_DIR = Path("artifacts/tables")

# ── colour palette (model → hex) ────────────────────────────────────────────
MODEL_COLORS: dict[str, str] = {
    "var_hist_sim": "#6c757d",
    "var_ewma": "#fd7e14",
    "var_garch_n": "#0d6efd",
    "var_garch_t": "#6f42c1",
    "var_lgbm_raw": "#20c997",
    "var_lgbm_cqr": "#ffc107",
    "var_lgbm_aci": "#dc3545",
    "var_lgbm_vol_aci": "#198754",
}
MODEL_LABELS: dict[str, str] = {
    "var_hist_sim": "Hist Sim (252d)",
    "var_ewma": "RiskMetrics EWMA",
    "var_garch_n": "GARCH-Normal",
    "var_garch_t": "GARCH-t",
    "var_lgbm_raw": "LightGBM (raw)",
    "var_lgbm_cqr": "LightGBM + CQR",
    "var_lgbm_aci": "LightGBM + ACI",
    "var_lgbm_vol_aci": "LightGBM + Vol-ACI",
}
REGIME_LABELS: dict[str, str] = {
    "full_sample_10y": "Full Sample (10y)",
    "calm_bull": "Calm Bull (2017–18)",
    "ilfs_liquidity_crisis": "IL&FS Crisis (2018–19)",
    "covid_crash": "COVID Crash (2020)",
    "rate_hike_tightening": "Rate Hike (2022)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_predictions(ticker_key: str, alpha: float) -> pd.DataFrame | None:
    tag = f"alpha{int(alpha * 100):03d}"
    path = DEMO_DIR / f"{ticker_key}_predictions_{tag}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False)
def load_portfolio_returns() -> pd.DataFrame | None:
    path = DEMO_DIR / "portfolio_universe_returns.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False)
def load_regime_scorecard() -> pd.DataFrame | None:
    path = TABLE_DIR / "regime_scorecard.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _data_as_of(df: pd.DataFrame) -> str:
    return df.index[-1].strftime("%d %b %Y")


def _not_ready_warning() -> None:
    st.warning(
        "⚠️ Precomputed demo data not found. "
        "Run `python scripts/precompute_demo.py` once to generate it.",
        icon="⚠️",
    )


def _compute_metrics(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Build summary metrics table from a predictions DataFrame."""
    rows = []
    for vc, label in MODEL_LABELS.items():
        if vc not in df.columns:
            continue
        fc = vc.replace("var_", "violation_")
        viols = df[fc].values if fc in df.columns else (df["realized_return"] < df[vc]).values
        n = len(viols)
        n_v = int(viols.sum())
        emp_alpha = n_v / n
        cov = 1.0 - emp_alpha
        width = float(np.mean(np.abs(df[vc].values)))
        r = df["realized_return"].values
        v = df[vc].values
        pinball = float(np.mean((alpha - (r < v).astype(float)) * (r - v)))
        rows.append(
            {
                "Model": label,
                "Coverage": f"{cov * 100:.1f}%",
                "Violations": n_v,
                "Cov Error": f"{abs(emp_alpha - alpha) * 100:.2f}pp",
                "Width (mean |VaR|)": f"{width * 100:.2f}%",
                "Pinball Loss": f"{pinball:.6f}",
                "_cov_err": abs(emp_alpha - alpha),
                "_width": width,
                "_pinball": pinball,
            }
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/NSE_logo.svg/320px-NSE_logo.svg.png",
        width=120,
    )
    st.title("Conformal VaR India")
    st.caption(
        "Uncertainty-calibrated Value-at-Risk on Indian equities using "
        "Adaptive Conformal Inference (ACI) + LightGBM."
    )
    st.divider()
    st.markdown("**Links**")
    st.markdown(
        "- [GitHub](https://github.com/RealDarthVader/conformal-var-india)\n"
        "- [FastAPI /docs](/docs)\n"
        "- [README](https://github.com/RealDarthVader/conformal-var-india#readme)"
    )
    st.divider()
    st.caption(
        "All VaR estimates are derived outputs. No raw price data is redistributed. "
        "For research purposes only."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_crash, tab_compare, tab_port = st.tabs(
    ["🔴 Crash Replay", "📊 Model Comparison", "💼 Portfolio Calculator"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Crash Replay
# ═══════════════════════════════════════════════════════════════════════════════
with tab_crash:
    st.header("Crash Replay: VaR Band Behaviour During Market Stress")
    st.caption(
        "Shows how each model's VaR band and ACI's dynamic α_t evolve during a crash. "
        "Red dots are violation days (realized loss > VaR)."
    )

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        scenario = st.selectbox(
            "Scenario",
            options=["March 2020 COVID Crash", "YES Bank Collapse (2018–2020)"],
            index=0,
        )
    with col_b:
        alpha_cr = st.selectbox("Confidence level α", options=[0.05, 0.01], index=0, key="cr_alpha")
    with col_c:
        selected_models_cr = st.multiselect(
            "Models",
            options=list(MODEL_LABELS.keys()),
            default=["var_garch_t", "var_lgbm_aci", "var_lgbm_vol_aci"],
            format_func=lambda k: MODEL_LABELS[k],
            key="cr_models",
        )

    ticker_key = "nsei" if "March 2020" in scenario else "yesbank"
    pred_df = load_predictions(ticker_key, alpha_cr)

    if pred_df is None:
        _not_ready_warning()
    else:
        # Date filter
        if "March 2020" in scenario:
            start_dt, end_dt = "2020-01-02", "2020-07-31"
            title_suffix = "Nifty 50 · March 2020 COVID Crash"
        else:
            start_dt, end_dt = "2018-01-01", "2020-12-31"
            title_suffix = "YES Bank · Collapse 2018–2020"

        window = pred_df.loc[start_dt:end_dt]

        if window.empty:
            st.error("No data in the selected date range.")
        else:
            st.caption(f"Data as of {_data_as_of(pred_df)}")
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.72, 0.28],
                vertical_spacing=0.04,
                subplot_titles=[
                    f"{title_suffix} — Daily Log Returns & VaR Bands (α={alpha_cr})",
                    "ACI Dynamic α_t (adaptive miscoverage level)",
                ],
            )

            # Returns bars
            colors_ret = [
                "#dc3545" if r < 0 else "#198754"
                for r in window["realized_return"]
            ]
            fig.add_trace(
                go.Bar(
                    x=window.index,
                    y=window["realized_return"] * 100,
                    marker_color=colors_ret,
                    name="Daily return (%)",
                    opacity=0.65,
                    showlegend=True,
                ),
                row=1, col=1,
            )

            # VaR bands
            all_violations_x: list = []
            all_violations_y: list = []
            for vc in selected_models_cr:
                if vc not in window.columns:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=window.index,
                        y=window[vc] * 100,
                        mode="lines",
                        name=MODEL_LABELS[vc],
                        line={"color": MODEL_COLORS[vc], "width": 1.8},
                    ),
                    row=1, col=1,
                )
                # Violation markers for this model
                fc = vc.replace("var_", "violation_")
                viol_mask = (
                    window[fc].astype(bool) if fc in window.columns
                    else window["realized_return"] < window[vc]
                )
                if viol_mask.any():
                    all_violations_x.extend(window.index[viol_mask].tolist())
                    all_violations_y.extend((window["realized_return"][viol_mask] * 100).tolist())

            if all_violations_x:
                fig.add_trace(
                    go.Scatter(
                        x=all_violations_x,
                        y=all_violations_y,
                        mode="markers",
                        name="Violation",
                        marker={"color": "red", "size": 7, "symbol": "circle", "line": {"color": "darkred", "width": 1}},
                        showlegend=True,
                    ),
                    row=1, col=1,
                )

            # ACI alpha_t path
            if "aci_alpha_t" in window.columns:
                fig.add_trace(
                    go.Scatter(
                        x=window.index,
                        y=window["aci_alpha_t"],
                        mode="lines",
                        name="ACI α_t",
                        line={"color": "#dc3545", "width": 1.5, "dash": "solid"},
                        showlegend=False,
                    ),
                    row=2, col=1,
                )
                # Nominal alpha line
                fig.add_hline(
                    y=alpha_cr, row=2, col=1,
                    line={"color": "grey", "dash": "dash", "width": 1},
                    annotation_text=f"Nominal α={alpha_cr}",
                    annotation_position="bottom right",
                )

            fig.update_yaxes(title_text="Log Return (%)", row=1, col=1)
            fig.update_yaxes(title_text="α_t", row=2, col=1)
            fig.update_layout(
                height=600,
                hovermode="x unified",
                legend={"orientation": "h", "y": -0.12},
                margin={"t": 60, "b": 20},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Quick violation count card
            st.divider()
            st.subheader("Violation summary over selected window")
            viol_rows = []
            for vc, label in MODEL_LABELS.items():
                if vc not in window.columns:
                    continue
                fc = vc.replace("var_", "violation_")
                n_v = int(window[fc].sum()) if fc in window.columns else int((window["realized_return"] < window[vc]).sum())
                emp = n_v / len(window)
                viol_rows.append({"Model": label, "Violations": n_v, "Empirical α": f"{emp * 100:.1f}%", "Nominal α": f"{alpha_cr * 100:.0f}%"})
            st.dataframe(pd.DataFrame(viol_rows).set_index("Model"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.header("Model Comparison: Coverage vs Width Trade-off")
    st.caption(
        "Coverage error (|empirical α − nominal α|) on the X-axis vs mean |VaR| band width on Y. "
        "Better models are toward the bottom-left. Showing GARCH-t winning on pinball loss is a "
        "credibility signal, not a problem."
    )

    col_d, col_e = st.columns([2, 1])
    with col_d:
        asset_cmp = st.selectbox("Asset", ["^NSEI (Nifty 50)", "YESBANK.NS"], index=0, key="cmp_asset")
    with col_e:
        alpha_cmp = st.selectbox("α", [0.05, 0.01], index=0, key="cmp_alpha")

    cmp_key = "nsei" if "NSEI" in asset_cmp else "yesbank"
    cmp_df = load_predictions(cmp_key, alpha_cmp)
    scorecard = load_regime_scorecard()

    if cmp_df is None:
        _not_ready_warning()
    else:
        st.caption(f"Full OOS period · data as of {_data_as_of(cmp_df)}")
        metrics_df = _compute_metrics(cmp_df, alpha_cmp)

        # ── Scatter: coverage error vs width ─────────────────────────────────
        fig_sc = go.Figure()
        for _, row in metrics_df.iterrows():
            vc_key = next((k for k, v in MODEL_LABELS.items() if v == row["Model"]), None)
            color = MODEL_COLORS.get(vc_key, "#888") if vc_key else "#888"
            fig_sc.add_trace(
                go.Scatter(
                    x=[row["_cov_err"] * 100],
                    y=[row["_width"] * 100],
                    mode="markers+text",
                    name=row["Model"],
                    text=[row["Model"]],
                    textposition="top center",
                    marker={"size": 16, "color": color, "line": {"width": 1.5, "color": "white"}},
                    customdata=[[row["Coverage"], row["Violations"], row["Pinball Loss"]]],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Coverage: %{customdata[0]}<br>"
                        "Violations: %{customdata[1]}<br>"
                        "Pinball: %{customdata[2]}<br>"
                        "Cov Error: %{x:.2f}pp<br>"
                        "Width: %{y:.3f}%"
                        "<extra></extra>"
                    ),
                )
            )
        fig_sc.update_layout(
            xaxis_title="Coverage Error (|empirical α − nominal α|) in pp",
            yaxis_title="Mean VaR Band Width (%)",
            height=420,
            showlegend=False,
            margin={"t": 30, "b": 40},
            hovermode="closest",
        )
        # ideal corner annotation
        fig_sc.add_annotation(
            x=0, y=0,
            text="← ideal (narrow & accurate)",
            showarrow=False,
            xanchor="left",
            font={"color": "grey", "size": 11},
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # ── Metrics table ─────────────────────────────────────────────────────
        st.subheader("Full Metrics Table")
        display = metrics_df[["Model", "Coverage", "Violations", "Cov Error", "Width (mean |VaR|)", "Pinball Loss"]].copy()

        def _style_coverage(val: str) -> str:
            try:
                v = float(val.rstrip("%"))
                target = (1 - alpha_cmp) * 100
                if abs(v - target) <= 0.5:
                    return "color: #198754; font-weight: bold"
                if abs(v - target) > 2:
                    return "color: #dc3545"
            except ValueError:
                pass
            return ""

        st.dataframe(
            display.style.applymap(_style_coverage, subset=["Coverage"]),
            use_container_width=True,
            hide_index=True,
        )

        # ── Regime drill-down from precomputed scorecard ──────────────────────
        if scorecard is not None:
            st.divider()
            st.subheader("Regime-Level Drill-down (from precomputed scorecard)")
            ticker_filter = "^NSEI" if "NSEI" in asset_cmp else "YESBANK.NS"
            sc_filtered = scorecard[
                (scorecard["Ticker"] == ticker_filter) & (scorecard["Alpha"] == alpha_cmp)
            ].copy()

            if not sc_filtered.empty:
                sc_filtered["Regime"] = sc_filtered["Regime"].map(REGIME_LABELS).fillna(sc_filtered["Regime"])

                regime_order = list(REGIME_LABELS.values())
                sc_filtered["_regime_sort"] = sc_filtered["Regime"].map(
                    {r: i for i, r in enumerate(regime_order)}
                ).fillna(99)
                sc_filtered = sc_filtered.sort_values(["_regime_sort", "Model"])

                regime_select = st.selectbox(
                    "Filter by regime",
                    ["All regimes"] + [r for r in regime_order if r in sc_filtered["Regime"].values],
                    key="regime_select",
                )
                if regime_select != "All regimes":
                    sc_filtered = sc_filtered[sc_filtered["Regime"] == regime_select]

                cols_to_show = ["Regime", "Model", "Coverage", "Sharpness", "Pinball Loss", "Kupiec p-val", "Basel Zone"]
                available_cols = [c for c in cols_to_show if c in sc_filtered.columns]
                st.dataframe(sc_filtered[available_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

        # ── Pinball bar chart ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Pinball Loss by Model (lower = better)")
        fig_pb = go.Figure(
            go.Bar(
                x=metrics_df["Model"],
                y=metrics_df["_pinball"],
                marker_color=[
                    MODEL_COLORS.get(next((k for k, v in MODEL_LABELS.items() if v == m), ""), "#888")
                    for m in metrics_df["Model"]
                ],
                text=[f"{v:.6f}" for v in metrics_df["_pinball"]],
                textposition="outside",
            )
        )
        fig_pb.update_layout(
            yaxis_title="Pinball Loss",
            height=320,
            margin={"t": 20, "b": 60},
            xaxis={"tickangle": -30},
        )
        st.plotly_chart(fig_pb, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Portfolio Calculator
# ═══════════════════════════════════════════════════════════════════════════════
with tab_port:
    st.header("Portfolio VaR & ES Calculator")
    st.caption(
        "Select stocks from the precomputed universe, set weights, and get "
        "1-day-ahead VaR and Expected Shortfall in ₹. "
        "All computation uses cached historical returns — no live data fetched."
    )

    port_returns = load_portfolio_returns()

    if port_returns is None:
        _not_ready_warning()
    else:
        st.caption(f"Universe: {len(port_returns.columns)} tickers · data as of {_data_as_of(port_returns)}")

        ticker_display = {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services",
            "HDFCBANK.NS": "HDFC Bank",
            "INFY.NS": "Infosys",
            "ICICIBANK.NS": "ICICI Bank",
            "SBIN.NS": "State Bank of India",
            "BHARTIARTL.NS": "Bharti Airtel",
            "ITC.NS": "ITC Limited",
            "LT.NS": "Larsen & Toubro",
            "KOTAKBANK.NS": "Kotak Mahindra Bank",
            "YESBANK.NS": "Yes Bank ⚠️ (stressed)",
            "IDEA.NS": "Vodafone Idea ⚠️ (stressed)",
            "ZEEL.NS": "Zee Entertainment ⚠️ (stressed)",
        }
        available = [t for t in port_returns.columns if t in ticker_display]

        col_p1, col_p2 = st.columns([3, 2])
        with col_p1:
            selected_tickers = st.multiselect(
                "Select stocks",
                options=available,
                default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"],
                format_func=lambda t: f"{t}  ({ticker_display.get(t, t)})",
            )
        with col_p2:
            alpha_p = st.selectbox("Confidence level α", [0.05, 0.01], index=1, key="port_alpha")
            port_value = st.number_input(
                "Portfolio Value (₹)",
                min_value=100_000,
                max_value=100_000_000,
                value=10_000_000,
                step=100_000,
                format="%d",
            )

        if not selected_tickers:
            st.info("Select at least one stock.")
        else:
            # Weight inputs
            st.subheader("Set Weights")
            st.caption("Weights are automatically normalised to sum to 1.")
            weight_cols = st.columns(min(len(selected_tickers), 4))
            raw_weights: dict[str, float] = {}
            for i, tkr in enumerate(selected_tickers):
                with weight_cols[i % 4]:
                    default_w = round(1.0 / len(selected_tickers), 3)
                    raw_weights[tkr] = st.number_input(
                        f"{tkr.split('.')[0]}",
                        min_value=0.0,
                        max_value=1.0,
                        value=default_w,
                        step=0.05,
                        key=f"w_{tkr}",
                    )

            total_w = sum(raw_weights.values())
            if total_w <= 0:
                st.error("At least one weight must be positive.")
            else:
                weights = {t: w / total_w for t, w in raw_weights.items()}

                # Portfolio log-return series (log of weighted price sum)
                sub_ret = port_returns[selected_tickers].dropna(how="all")
                # Weighted sum of price-relatives then log
                price_rel = np.exp(sub_ret.fillna(0.0))
                port_pr = price_rel.dot(pd.Series(weights))
                port_ret = np.log(port_pr)

                window_ret = port_ret.dropna()

                if len(window_ret) < 500:
                    st.error(f"Need at least 500 days of history. Got {len(window_ret)}.")
                else:
                    # ACI-calibrated HS VaR using last 500 days as rolling basis
                    recent = window_ret.iloc[-500:]

                    # Historical Simulation VaR
                    hs_var = float(np.quantile(recent, alpha_p))

                    # ACI: use last 252 as calibration buffer
                    calib_ret = window_ret.iloc[-252:]
                    hs_rolling = [
                        float(np.quantile(window_ret.iloc[max(0, i - 252):i], alpha_p))
                        for i in range(len(window_ret) - 252, len(window_ret))
                    ]
                    residuals = np.array(hs_rolling) - calib_ret.values
                    m = len(residuals)
                    import math
                    level = min(math.ceil((m + 1) * (1 - alpha_p)) / m, 1.0)
                    q_conf = float(np.quantile(residuals, level))
                    aci_var = hs_var - q_conf

                    # Expected Shortfall: mean of returns below VaR
                    tail_mask = recent.values < aci_var
                    es = float(np.mean(recent.values[tail_mask])) if tail_mask.any() else aci_var * 1.3

                    # INR amounts
                    var_inr = abs(aci_var) * port_value
                    es_inr = abs(es) * port_value

                    # 250-day backtest
                    bt_250 = window_ret.iloc[-250:].values
                    viols_250 = (bt_250 < hs_var).astype(int)
                    emp_alpha_250 = float(viols_250.mean())
                    n_v_250 = int(viols_250.sum())
                    # Basel zone
                    if n_v_250 <= 4:
                        zone, zone_color = "🟢 GREEN", "#198754"
                    elif n_v_250 <= 9:
                        zone, zone_color = "🟡 YELLOW", "#ffc107"
                    else:
                        zone, zone_color = "🔴 RED", "#dc3545"

                    st.divider()
                    st.subheader("📐 VaR & ES Estimates (ACI-calibrated)")

                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("1-Day VaR (return)", f"{aci_var * 100:.3f}%")
                    kpi2.metric("1-Day VaR (₹)", f"₹{var_inr:,.0f}")
                    kpi3.metric("Expected Shortfall (return)", f"{es * 100:.3f}%")
                    kpi4.metric("Expected Shortfall (₹)", f"₹{es_inr:,.0f}")

                    st.caption(
                        f"α = {alpha_p} · Portfolio value ₹{port_value:,} · "
                        f"Normalised weights: "
                        + ", ".join(f"{t.split('.')[0]}: {w * 100:.1f}%" for t, w in weights.items())
                    )

                    # Basel 250-day backtest
                    st.divider()
                    st.subheader("🏦 250-Day Basel Backtest")
                    bk1, bk2, bk3 = st.columns(3)
                    bk1.metric("Violations (last 250d)", n_v_250)
                    bk2.metric("Empirical α", f"{emp_alpha_250 * 100:.1f}%", delta=f"target: {alpha_p * 100:.0f}%")
                    bk3.metric("Basel Zone", zone)

                    # Portfolio return histogram with VaR line
                    st.divider()
                    st.subheader("Return Distribution (last 500 trading days)")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(
                        go.Histogram(
                            x=recent.values * 100,
                            nbinsx=60,
                            name="Daily Returns",
                            marker_color="#0d6efd",
                            opacity=0.7,
                        )
                    )
                    fig_hist.add_vline(
                        x=aci_var * 100,
                        line={"color": "#dc3545", "dash": "dash", "width": 2},
                        annotation_text=f"ACI VaR ({alpha_p*100:.0f}%): {aci_var*100:.2f}%",
                        annotation_position="top left",
                    )
                    fig_hist.add_vline(
                        x=es * 100,
                        line={"color": "#fd7e14", "dash": "dot", "width": 2},
                        annotation_text=f"ES: {es*100:.2f}%",
                        annotation_position="top left",
                    )
                    fig_hist.update_layout(
                        xaxis_title="Daily Log Return (%)",
                        yaxis_title="Count",
                        height=320,
                        margin={"t": 20, "b": 40},
                        showlegend=False,
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    # Cumulative portfolio return
                    st.subheader("Cumulative Portfolio Return (full history)")
                    cum = (window_ret + 1).cumprod()
                    fig_cum = go.Figure(
                        go.Scatter(
                            x=window_ret.index,
                            y=(cum - 1) * 100,
                            fill="tozeroy",
                            fillcolor="rgba(13,110,253,0.12)",
                            line={"color": "#0d6efd", "width": 1.5},
                            name="Cumulative Return (%)",
                        )
                    )
                    fig_cum.update_layout(
                        yaxis_title="Cumulative Return (%)",
                        height=280,
                        margin={"t": 10, "b": 30},
                        hovermode="x",
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Conformal VaR India** · Research prototype · "
    "Not investment advice · "
    "[GitHub](https://github.com/RealDarthVader/conformal-var-india)"
)
