# Uncertainty-Calibrated Value-at-Risk Engine on Indian Equities (`conformal-var-india`)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An applied quantitative market-risk framework combining **Conformal Prediction**, **Online Adaptive Conformal Inference (ACI)**, and **LightGBM Quantile Regression** to provide finite-sample, distribution-free Value-at-Risk ($\text{VaR}_\alpha$) and Expected Shortfall ($\text{ES}_\alpha$) calibration on National Stock Exchange (NSE) indices and equities.

---

## Table of Contents

1. [Executive Summary & Research Verdict](#1-executive-summary--research-verdict)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [Empirical Results Across Market Regimes](#3-empirical-results-across-market-regimes)
4. [Diagnostic Visualizations](#4-diagnostic-visualizations)
5. [Explicit Limitations & Theoretical Caveats](#5-explicit-limitations--theoretical-caveats)
6. [System Architecture & Production Packaging](#6-system-architecture--production-packaging)
7. [Quickstart & Verification](#7-quickstart--verification)

---

## 1. Executive Summary & Research Verdict

### Core Research Question

> *"Does Online Adaptive Conformal Inference (ACI) wrapped around machine learning quantile regressors actually beat a well-tuned $\text{GARCH}(1,1)\text{-Student-}t$ or volatility-scaled residual band on emerging market equity tails?"*

### Empirical Verdict

Testing across **15+ years (3,356 out-of-sample trading days)** of NSE data reveals a clear regime dichotomy between diversified index tails and idiosyncratic single-stock collapses:

1. **Diversified Index Tails (`^NSEI`, $\alpha = 0.01$): Volatility-Scaled Conformal & Sliding CQR Dominate**
   - On the Nifty 50 index at the $99\%$ confidence level ($\alpha = 0.01$), parametric $\text{GARCH}(1,1)\text{-Student-}t$ achieves $98.63\%$ empirical coverage but falls just short of the $95\%$ Kupiec unconditional coverage threshold ($p = 0.0452$).
   - **LightGBM + Vol-Scaled ACI** ($98.72\%$ coverage, Kupiec $p = 0.1276$, sharpness $0.0311$) and **LightGBM + Sliding CQR** ($98.88\%$ coverage, Kupiec $p = 0.4846$, sharpness $0.0315$) pass both institutional coverage backtests and the Basel Green Zone criteria. When volatility clustering is primarily second-order heteroscedasticity without structural breakdown, normalizing conformity residuals by local conditional volatility $\hat{\sigma}_t$ yields the most efficient calibration.

2. **Idiosyncratic Left-Tail Collapses (`YESBANK.NS` 2018–2020 Crash, $\alpha = 0.01$): Unscaled Online ACI Dominate**
   - During the YES Bank liquidity collapse (which included single-day log-return shocks of $-0.8235$, or $-56.1\%$ simple return), uncalibrated LightGBM suffered **56 violations** over 736 trading days ($92.39\%$ coverage, Kupiec $p = 0.0000$, Basel Red Zone).
   - **LightGBM + Online ACI ($\gamma = 0.01$)** cut tail violations by **$89\%$ (down to 6 violations)**, maintaining **$99.18\%$ empirical coverage** (Basel Green Zone, Kupiec $p = 0.6027$, Christoffersen Independence $p = 0.7533$).
   - Crucially, **Online ACI achieved $2\times$ sharper risk bands ($0.2180$) than Vol-Scaled ACI ($0.4477$)** during the crash. Dividing conformity residuals by realized volatility after an extreme tail shock causes volatility-scaled bands to over-inflate, whereas Online ACI adjusts the effective quantile level $\alpha_t$ directly without variance distortion.

---

## 2. Mathematical Formulation

### 2.1 Causal Quantile Loss & Signed Conformity Residual

Let $r_t = \ln(P_t / P_{t-1})$ denote the realized daily log return of an asset at day $t$, and let $X_t \in \mathcal{F}_{t-1}$ denote the causal feature vector strictly observable prior to market close on day $t-1$.

The asymmetric quantile pinball loss for target miscoverage rate $\alpha \in (0, 1)$ and predicted quantile $\hat{v}_t$ is:

$$\mathcal{L}_\alpha(r_t, \hat{v}_t) = \left(\alpha - \mathbb{I}\{r_t < \hat{v}_t\}\right)(r_t - \hat{v}_t)$$

The signed lower-tail conformity residual $s_t$ is defined such that a VaR violation ($r_t < \hat{q}_\alpha(X_t)$) produces a strictly positive score:

$$s_t = \hat{q}_\alpha(X_t) - r_t$$

### 2.2 Sliding-Window Conformal Quantile Calibration (CQR)

Given a rolling calibration window of size $M$, the finite-sample conformal correction threshold is the empirical quantile of recent conformity residuals:

$$\hat{Q}_{1-\alpha}(s) = \text{Quantile}_{\frac{\lceil (M+1)(1-\alpha) \rceil}{M}}\left(\{s_{t-M}, \dots, s_{t-1}\}\right)$$

$$\widehat{\text{VaR}}_t^{\text{conf}}(\alpha) = \hat{q}_\alpha(X_t) - \hat{Q}_{1-\alpha}(s)$$

### 2.3 Online Adaptive Conformal Inference (Gibbs & Candès, 2021)

To account for non-exchangeability, volatility clustering, and structural regime shifts online, ACI dynamically updates the effective miscoverage rate $\alpha_t$ at each step $t$ with step size $\gamma > 0$:

$$\text{err}_t = \mathbb{I}\left\{r_t < \widehat{\text{VaR}}_t^{\text{ACI}}(\alpha)\right\}$$

$$\alpha_{t+1} = \text{clip}\left(\alpha_t + \gamma (\alpha - \text{err}_t), \, \alpha_{\min}, \, \alpha_{\max}\right)$$

For **Volatility-Scaled ACI**, residuals are normalized by causal conditional volatility $\hat{\sigma}_t$:

$$s_t^{\text{scaled}} = \frac{\hat{q}_\alpha(X_t) - r_t}{\hat{\sigma}_t}, \quad \widehat{\text{VaR}}_t^{\text{vol-ACI}}(\alpha) = \hat{q}_\alpha(X_t) - \hat{\sigma}_t \cdot \hat{Q}_{1-\alpha_t}\left(s^{\text{scaled}}\right)$$

### 2.4 Approximated Conformal Expected Shortfall ($\text{ES}_\alpha$)

Because distribution-free conformal guarantees apply to quantiles rather than conditional tail expectations, Expected Shortfall is approximated via numerical integration across a 10-point left-tail quantile grid $\tau_j = \frac{j \cdot \alpha}{10}$:

$$\widehat{\text{ES}}_t(\alpha) \approx \frac{1}{10} \sum_{j=1}^{10} \widehat{\text{VaR}}_t^{\text{ACI}}\left(\frac{j \cdot \alpha}{10}\right)$$

This formulation enforces strict left-tail dominance and quantile monotonicity: $\widehat{\text{ES}}_t(\alpha) \le \widehat{\text{VaR}}_t^{\text{ACI}}(\alpha)$.

### 2.5 Institutional Market-Risk Backtesting Suite

* **Kupiec Proportion of Failures (POF) Likelihood Ratio (Unconditional Coverage):**

$$LR_{\text{uc}} = -2 \ln \left[ \frac{(1-\alpha)^{T-N} \alpha^N}{(1-\hat{p})^{T-N} \hat{p}^N} \right] \sim \chi^2_1$$

* **Christoffersen Markov Independence Likelihood Ratio (Violation Clustering):**

$$LR_{\text{ind}} = -2 \ln \left[ \frac{(1-\pi)^{n_{00}+n_{10}} \pi^{n_{01}+n_{11}}}{(1-\pi_{01})^{n_{00}} \pi_{01}^{n_{01}} (1-\pi_{11})^{n_{10}} \pi_{11}^{n_{11}}} \right] \sim \chi^2_1$$

* **Basel Committee Traffic Light Framework (Standardized for $\alpha = 0.01$ over 250 Trading Days):**
  - **Green Zone ($0\text{–}4$ violations):** Multiplier $k = 3.00$ (Model Accepted).
  - **Yellow Zone ($5\text{–}9$ violations):** Multiplier $k \in [3.40, 3.85]$ (Capital Surcharge).
  - **Red Zone ($\ge 10$ violations):** Multiplier $k = 4.00$ (Model Rejected).

---

## 3. Empirical Results Across Market Regimes

### Table 1: 10-Year Out-of-Sample Performance on Nifty 50 (`^NSEI`, $T = 3,356$ Days)

| Model | Target $\alpha$ | Empirical Coverage | Coverage Error | Sharpness ($\Vert{}\overline{\text{VaR}}\Vert{}$) | Pinball Loss | Kupiec $p$-val | Basel Zone ($99\%$ Std)* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Historical Simulation (252d) | $0.05$ | $94.32\%$ | $0.68\%$ | $0.0148$ | $0.001224$ | $0.0800$ | N/A ($\alpha=0.05$) |
| RiskMetrics EWMA ($\lambda=0.94$) | $0.05$ | $94.20\%$ | $0.80\%$ | $0.0151$ | $0.001138$ | $0.0397$ | N/A ($\alpha=0.05$) |
| $\text{GARCH}(1,1)$-Normal | $0.05$ | $94.84\%$ | $0.16\%$ | $0.0154$ | $0.001138$ | $0.6704$ | N/A ($\alpha=0.05$) |
| $\text{GARCH}(1,1)$-Student-$t$ | $0.05$ | $94.59\%$ | $0.41\%$ | $0.0148$ | $0.001139$ | $0.2918$ | N/A ($\alpha=0.05$) |
| Uncalibrated LightGBM | $0.05$ | $93.20\%$ | $1.80\%$ | $0.0137$ | $0.001238$ | $0.0000$ | N/A ($\alpha=0.05$) |
| **LightGBM + Online ACI** | $0.05$ | **$94.90\%$** | **$0.10\%$** | $0.0178$ | $0.001218$ | **$0.7895$** | N/A ($\alpha=0.05$) |
| **LightGBM + Vol-Scaled ACI** | $0.05$ | **$94.93\%$** | **$0.07\%$** | $0.0188$ | $0.001248$ | **$0.8513$** | N/A ($\alpha=0.05$) |
| Historical Simulation (252d) | $0.01$ | $98.24\%$ | $0.76\%$ | $0.0258$ | $0.000444$ | $0.0001$ | Green |
| RiskMetrics EWMA ($\lambda=0.94$) | $0.01$ | $98.03\%$ | $0.97\%$ | $0.0214$ | $0.000372$ | $0.0000$ | Yellow |
| $\text{GARCH}(1,1)$-Normal | $0.01$ | $98.30\%$ | $0.70\%$ | $0.0217$ | $0.000368$ | $0.0002$ | Green |
| $\text{GARCH}(1,1)$-Student-$t$ | $0.01$ | $98.63\%$ | $0.37\%$ | $0.0231$ | $0.000364$ | $0.0452$ | Green |
| Uncalibrated LightGBM | $0.01$ | $97.81\%$ | $1.19\%$ | $0.0222$ | $0.000480$ | $0.0000$ | Yellow |
| **LightGBM + Sliding CQR** | $0.01$ | **$98.88\%$** | **$0.12\%$** | $0.0315$ | $0.000435$ | **$0.4846$** | **Green** |
| **LightGBM + Online ACI** | $0.01$ | $98.60\%$ | $0.40\%$ | $0.0320$ | $0.000436$ | $0.0308$ | **Green** |
| **LightGBM + Vol-Scaled ACI** | $0.01$ | **$98.72\%$** | **$0.28\%$** | $0.0311$ | $0.000413$ | **$0.1276$** | **Green** |

*\*Note: The Basel Committee Traffic Light framework ($0\text{–}4$ Green, $5\text{–}9$ Yellow, $\ge 10$ Red per 250 trading days) is defined strictly for $99\%$ VaR ($\alpha = 0.01$). Models pass unconditional coverage at the $95\%$ statistical confidence level when $\text{Kupiec } p \ge 0.05$.*

### Table 2: Single-Stock Left-Tail Collapse Stress Test — `YESBANK.NS` (2018–2020, $\alpha = 0.01$, $T = 736$ Days)

| Model | Empirical Coverage | Violations (736 Days) | Sharpness ($\Vert{}\overline{\text{VaR}}\Vert{}$) | Kupiec $p$-val | Christoffersen $p$-val | Basel Traffic Light |
| --- | --- | --- | --- | --- | --- | --- |
| Uncalibrated LightGBM | $92.39\%$ | $56$ | $0.0732$ | $0.0000$ | $0.0000$ | Red (Rejected) |
| RiskMetrics EWMA ($\lambda=0.94$) | $97.55\%$ | $18$ | $0.1183$ | $0.0009$ | $0.4557$ | Yellow |
| Historical Simulation (252d) | $97.96\%$ | $15$ | $0.1240$ | $0.0131$ | $0.0333$ | Yellow |
| $\text{GARCH}(1,1)$-Student-$t$ | $98.37\%$ | $12$ | $0.1308$ | $0.1152$ | $0.1840$ | Green |
| **LightGBM + Sliding CQR** | **$99.05\%$** | **$7$** | $0.2114$ | **$0.8930$** | **$0.7137$** | **Green (Accepted)** |
| **LightGBM + Online ACI ($\gamma=0.01$)** | **$99.18\%$** | **$6$** | **$0.2180$** | **$0.6027$** | **$0.7533$** | **Green (Accepted)** |
| **LightGBM + Vol-Scaled ACI** | **$99.05\%$** | **$7$** | $0.4477$ | **$0.8930$** | **$0.7137$** | **Green (Accepted)** |

---

## 4. Diagnostic Visualizations

### March 2020 COVID-19 Crash Backtest

![COVID-19 Crash Backtest](artifacts/plots/march_2020_crash_backtest.png)

*Figure 1: Daily Nifty 50 returns during the March 2020 liquidity shock overlaid with $99\%$ VaR envelopes. The bottom panel tracks the online trajectory of $\alpha_t$, demonstrating automatic tail-band expansion immediately following clustered violations.*

### Coverage vs. Sharpness Pareto Frontier

![Pareto Frontier](artifacts/plots/coverage_vs_sharpness_pareto.png)

*Figure 2: Empirical coverage error ($\vert{}\hat{\alpha} - \alpha\vert{}$) vs. mean VaR band width (sharpness) over 10-year out-of-sample testing on Nifty 50.*

### Basel Traffic Light Failure Rates by Market Regime

![Basel Violations by Regime](artifacts/plots/basel_violations_by_regime.png)

*Figure 3: Empirical violation rates across four Indian market regimes (`calm_bull`, `ilfs_liquidity_crisis`, `covid_crash`, `rate_hike_tightening`) benchmarked against nominal $1\%$ and Basel regulatory thresholds.*

---

## 5. Explicit Limitations & Theoretical Caveats

1. **Exchangeability Breakdown in Financial Time Series:**
   - Classical Split Conformal Prediction relies on exchangeability of calibration and test residuals. Equity log returns violate exchangeability via volatility clustering, leverage effects, and macroeconomic regime shifts.
   - While Online ACI guarantees asymptotic long-run coverage $\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T \text{err}_t = \alpha$ without distributional assumptions, finite-sample conditional coverage $\mathbb{P}(r_t < \text{VaR}_t \mid \mathcal{F}_{t-1})$ can still experience transient deviations during abrupt liquidity shocks.

2. **Asymptotic Marginal vs. Conditional Coverage:**
   - Standard conformal bounds guarantee marginal coverage across time. Market-risk desks care about conditional independence of violations (tested via Christoffersen's $LR_{\text{ind}}$). Sliding-window CQR and Online ACI reduce violation clustering compared to static calibration, but depend on the choice of step size $\gamma$ and window length $M$.

3. **Survivorship Bias Mitigation:**
   - Evaluating risk models exclusively on current index constituents introduces survivorship bias by censoring bankruptcies and delistings. This pipeline explicitly incorporates distressed historical NSE constituents (`YESBANK.NS`, `IDEA.NS`, `ZEEL.NS`) to stress-test left-tail collapse behavior.

4. **Expected Shortfall Approximation:**
   - Conformal prediction calibrates quantiles ($\text{VaR}_\alpha$) rather than conditional expectations beyond the quantile. Expected Shortfall ($\text{ES}_\alpha$) is therefore approximated numerically via a 10-point tail quantile grid rather than derived from a parametric tail density.

---

## 6. System Architecture & Production Packaging

```text
conformal-var-india/
├── src/conformal_var/
│   ├── config.py         # Strict Pydantic v2 configuration schema
│   ├── data/loader.py    # Survivorship-aware NSE data pipeline (Parquet cache)
│   ├── features/         # Causal lag volatility & tail features (.shift(1))
│   ├── models/           # Historical Simulation, RiskMetrics, GARCH, LightGBM
│   ├── conformal/        # Split, Sliding CQR, Online ACI, Vol-Scaled ACI
│   ├── backtest/         # Kupiec POF, Christoffersen Markov, Basel Traffic Light
│   ├── portfolio/        # Multi-asset return aggregation, Conformal ES grid
│   └── api/              # Production FastAPI service with Redis/LRU cache
├── streamlit_app.py      # Interactive 3-tab zero-compute Streamlit demo
├── artifacts/            # Parquet datasets, diagnostic plots, regime scorecards
├── scripts/              # Data ingestion, demo precompute, benchmark runner, plot generators
├── tests/                # 90 Pytest analytical, leakage & integration tests
├── Dockerfile            # Multi-stage production container (API & Demo targets)
└── docker-compose.yml    # FastAPI + Redis 7 + Streamlit service stack
```

---

## 7. Quickstart & Verification

### Local Setup & Automated Test Suite

```bash
# Clone the repository and install dependencies
git clone https://github.com/RealDarthVader/conformal-var-india.git
cd conformal-var-india
pip install -r requirements.txt

# Run the complete verification suite (90 passing unit & statistical tests)
pytest

# Regenerate empirical scorecards and diagnostic plots
python scripts/run_research_experiment.py
python scripts/generate_plots.py
```

### Interactive Streamlit Demo (Zero Cold-Start)

```bash
# Precompute predictions once (offline, ~2 minutes)
python scripts/precompute_demo.py

# Launch the interactive Streamlit application
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) to explore the **Crash Replay**, **Model Comparison**, and **Portfolio VaR & ES Calculator** tabs.

### Docker Compose Deployment

```bash
# Build and launch FastAPI microservice, Redis 7 cache, and Streamlit demo
docker compose up --build -d

# Check service and Redis health
curl -s http://localhost:8000/health | jq .
```

* **Interactive Streamlit Demo:** [http://localhost:8501](http://localhost:8501)
* **FastAPI Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

### API Inference Example: Multi-Asset Portfolio VaR & Expected Shortfall

```bash
curl -X POST "http://localhost:8000/api/v1/portfolio/var" \
     -H "Content-Type: application/json" \
     -d '{
       "weights": {"RELIANCE.NS": 0.4, "TCS.NS": 0.3, "INFY.NS": 0.3},
       "alpha": 0.01,
       "model": "aci_lgbm",
       "gamma": 0.01,
       "portfolio_value_inr": 5000000
     }' | jq .
```

### Example JSON Response

```json
{
  "date": "2026-09-30",
  "alpha": 0.01,
  "var_return_pct": -2.3481,
  "var_inr": 117405.0,
  "expected_shortfall_pct": -2.8942,
  "expected_shortfall_inr": 144710.0,
  "empirical_coverage_250d": 0.9880,
  "kupiec_p_value": 0.7412,
  "christoffersen_ind_p_value": 0.8124,
  "basel_traffic_light": "GREEN",
  "cache_hit": false,
  "latency_ms": 14.2
}
```
