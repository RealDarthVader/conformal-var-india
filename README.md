# Uncertainty-Calibrated Value-at-Risk Engine on Indian Equities (`conformal-var-india`)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An applied quantitative market-risk framework combining **Conformal Prediction**, **Adaptive Conformal Inference (ACI)**, and **LightGBM Quantile Regression** to provide finite-sample, distribution-free Value-at-Risk ($\text{VaR}_\alpha$) and Expected Shortfall ($\text{ES}_\alpha$) guarantees on National Stock Exchange (NSE) indices and equities.

---

## 📑 Table of Contents
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
> *"Does Online Adaptive Conformal Inference (ACI) on machine learning quantile regressors actually beat a well-tuned $\text{GARCH}(1,1)\text{-}t$ or volatility-scaled residual band on emerging market equity tails?"*

### Findings & Empirical Verdict
1. **When does Online ACI beat GARCH(1,1)-$t$?**
   - **Severe Left-Tail Collapses & Regime Discontinuities**: During the **YES Bank collapse (2018–2020)** and the **COVID-19 Crash (March 2020)**, parametric $\text{GARCH}(1,1)\text{-}t$ models suffered from lag in conditional variance adaptation when encountering unprecedented return shocks (e.g. single-day drops of $-82.35\%$).
   - In contrast, **Online ACI ($\gamma=0.01$)** adapted rapidly: during the YES Bank collapse at $\alpha=0.01$, uncalibrated LightGBM had 56 violations (Basel **Red** zone; Kupiec $p = 0.0$), whereas **LightGBM + Online ACI** had only 6 violations, maintaining $\mathbf{99.18\%}$ coverage (Basel **Green** zone; Kupiec $p = 0.6027$).
2. **When is ACI equivalent to Volatility Scaling?**
   - **Stationary & Moderate-Volatility Regimes**: In the **Calm Bull (2017–2018)** and **Rate Tightening (2022)** regimes, normalizing conformity residuals by local volatility $\hat{\sigma}_t$ ($\text{Vol-Scaled Conformal VaR}$) achieves identical empirical coverage with slightly sharper (tighter) bands than unscaled ACI.
   - When volatility clustering is purely second-order heteroscedasticity without structural regime breakdown, explicit volatility normalization provides the most efficient score representation.

---

## 2. Mathematical Formulation

### 2.1 Causal Quantile Loss & Signed Conformity Residual
Let $r_t$ denote the realized daily log return of an asset at day $t$, and let $X_t$ denote the causal feature vector strictly observable at market close of day $t-1$.

The asymmetric quantile pinball loss for quantile $\alpha \in (0, 1)$ is:
$$\mathcal{L}_\alpha(r_t, \hat{v}_t) = (\alpha - \mathbb{I}\{r_t < \hat{v}_t\})(r_t - \hat{v}_t)$$

The signed lower-tail conformity residual is defined such that an empirical risk violation ($r_t < \hat{q}_\alpha(X_t)$) produces a strictly positive residual:
$$s_t = \hat{q}_\alpha(X_t) - r_t$$

### 2.2 Conformal Quantile Calibration
Given calibration window size $M$, finite-sample adjustment establishes the threshold:
$$\hat{Q}_{1-\alpha}(s) = \text{Quantile}_{\frac{\lceil (M+1)(1-\alpha) \rceil}{M}}\left(\{s_{t-M}, \dots, s_{t-1}\}\right)$$
$$\widehat{\text{VaR}}_t^{\text{conf}}(\alpha) = \hat{q}_\alpha(X_t) - \hat{Q}_{1-\alpha}(s)$$

### 2.3 Adaptive Conformal Inference (Gibbs & Candès, 2021)
To accommodate non-exchangeability and volatility shifts online, ACI dynamically updates the effective coverage level $\alpha_t$:
$$\text{err}_t = \mathbb{I}\left\{r_t < \widehat{\text{VaR}}_t^{\text{ACI}}(\alpha)\right\}$$
$$\alpha_{t+1} = \text{clip}\left(\alpha_t + \gamma (\alpha - \text{err}_t), \, \alpha_{\min}, \, \alpha_{\max}\right)$$

### 2.4 Approximated Conformal Expected Shortfall ($\text{ES}_\alpha$)
Because conformal guarantees apply to quantiles rather than conditional tail expectations, $\text{ES}_\alpha$ is approximated via numeric integration across a 10-point left-tail quantile grid:
$$\widehat{\text{ES}}_t(\alpha) \approx \frac{1}{10} \sum_{j=1}^{10} \widehat{\text{VaR}}_t^{\text{ACI}}\left(\frac{j \cdot \alpha}{10}\right)$$
The engine mathematically guarantees sub-additivity and dominance: $\widehat{\text{ES}}_t(\alpha) \le \widehat{\text{VaR}}_t^{\text{ACI}}(\alpha)$.

### 2.5 Bank Market-Risk Backtesting Likelihood Ratios
- **Kupiec POF Test (Unconditional Coverage)**:
  $$LR_{\text{uc}} = -2 \ln \left[ \frac{(1-\alpha)^{T-N} \alpha^N}{(1-\hat{p})^{T-N} \hat{p}^N} \right] \sim \chi^2_1$$
- **Christoffersen Markov Independence Test (Violation Clustering)**:
  $$LR_{\text{ind}} = -2 \ln \left[ \frac{(1-\pi)^{n_{00}+n_{10}} \pi^{n_{01}+n_{11}}}{(1-\pi_{01})^{n_{00}} \pi_{01}^{n_{01}} (1-\pi_{11})^{n_{10}} \pi_{11}^{n_{11}}} \right] \sim \chi^2_1$$
- **Basel Committee Traffic Light Framework (250 Days)**:
  - **Green Zone** ($\le 4$ violations): Multiplier $k = 3.00$ (Model Accepted).
  - **Yellow Zone** ($5 \dots 9$ violations): Multiplier $k \in [3.40, 3.85]$ (Capital Surcharge).
  - **Red Zone** ($\ge 10$ violations): Multiplier $k = 4.00$ (Model Rejected).

---

## 3. Empirical Results Across Market Regimes

### Table 1: 10-Year Out-of-Sample Performance on Nifty 50 (^NSEI, 3,356 Out-of-Sample Days)
| Model | Target $\alpha$ | Empirical Coverage | Coverage Error | Sharpness | Pinball Loss | Kupiec $p$ | Basel Zone |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Historical Simulation (252d) | 0.05 | 94.32% | 0.68% | 0.0148 | 0.001224 | 0.0800 | Red |
| RiskMetrics EWMA ($\lambda=0.94$) | 0.05 | 94.20% | 0.80% | 0.0151 | 0.001138 | 0.0397 | Red |
| GARCH(1,1)-Normal | 0.05 | 94.84% | 0.16% | 0.0154 | 0.001138 | 0.6704 | Red |
| GARCH(1,1)-Student-$t$ | 0.05 | 94.59% | 0.41% | 0.0148 | 0.001139 | 0.2918 | Red |
| Uncalibrated LightGBM | 0.05 | 93.20% | 1.80% | 0.0137 | 0.001238 | 0.0000 | Red |
| **LightGBM + Online ACI** | 0.05 | **94.90%** | **0.10%** | 0.0178 | 0.001218 | **0.7895** | Red |
| **LightGBM + Vol-Scaled ACI** | 0.05 | **94.93%** | **0.07%** | 0.0188 | 0.001248 | **0.8513** | Red |
| Historical Simulation (252d) | 0.01 | 98.24% | 0.76% | 0.0258 | 0.000444 | 0.0001 | Green |
| RiskMetrics EWMA ($\lambda=0.94$) | 0.01 | 98.03% | 0.97% | 0.0214 | 0.000372 | 0.0000 | Yellow |
| GARCH(1,1)-Normal | 0.01 | 98.30% | 0.70% | 0.0217 | 0.000368 | 0.0002 | Green |
| GARCH(1,1)-Student-$t$ | 0.01 | 98.63% | 0.37% | 0.0231 | 0.000364 | 0.0452 | Green |
| Uncalibrated LightGBM | 0.01 | 97.81% | 1.19% | 0.0222 | 0.000480 | 0.0000 | Yellow |
| **LightGBM + Sliding CQR** | 0.01 | **98.88%** | **0.12%** | 0.0315 | 0.000435 | **0.4846** | **Green** |
| **LightGBM + Online ACI** | 0.01 | **98.60%** | **0.40%** | 0.0320 | 0.000436 | **0.0308** | **Green** |
| **LightGBM + Vol-Scaled ACI** | 0.01 | **98.72%** | **0.28%** | 0.0311 | 0.000413 | **0.1276** | **Green** |

---

### Table 2: Left-Tail Collapse Stress Test: YESBANK.NS (2018–2020, $\alpha=0.01$)
| Model | Empirical Coverage | Violations (736 Days) | Sharpness | Kupiec $p$-val | Christoffersen $p$-val | Basel Zone |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Uncalibrated LightGBM | 92.39% | 56 | 0.0732 | 0.0000 | 0.0000 | **Red (Rejected)** |
| RiskMetrics EWMA (0.94) | 97.55% | 18 | 0.1183 | 0.0009 | 0.4557 | Yellow |
| Historical Simulation | 97.96% | 15 | 0.1240 | 0.0131 | 0.0333 | Yellow |
| GARCH(1,1)-Student-$t$ | 98.37% | 12 | 0.1308 | 0.1152 | 0.1840 | Green |
| **LightGBM + Sliding CQR** | **99.05%** | **7** | **0.2114** | **0.8930** | **0.7137** | **Green (Accepted)** |
| **LightGBM + Online ACI** | **99.18%** | **6** | **0.2180** | **0.6027** | **0.7533** | **Green (Accepted)** |
| **LightGBM + Vol-Scaled ACI** | **99.05%** | **7** | **0.4477** | **0.8930** | **0.7137** | **Green (Accepted)** |

---

## 4. Diagnostic Visualizations

### March 2020 COVID-19 Crash Backtest
![COVID-19 Crash Backtest](artifacts/plots/march_2020_crash_backtest.png)
*Figure 1: Daily Nifty 50 returns during the March 2020 liquidity shock. Notice the bottom subplot tracking the online adaptation of $\alpha_t$, expanding the VaR envelope dynamically after violations occur.*

### Coverage vs. Sharpness Pareto Frontier
![Pareto Frontier](artifacts/plots/coverage_vs_sharpness_pareto.png)
*Figure 2: Empirical coverage error vs. sharpness over 10-year out-of-sample testing on Nifty 50. Conformalized models dominate the low coverage error region.*

### Basel Traffic Light Failure Rates by Market Regime
![Basel Violations by Regime](artifacts/plots/basel_violations_by_regime.png)
*Figure 3: Empirical failure rate across the 4 Indian market regimes against Basel Green/Yellow/Red zone thresholds.*

---

## 5. Explicit Limitations & Theoretical Caveats

1. **Exchangeability Breakdown in Financial Time Series**:
   - Conformal Prediction theorems rely on exchangeable calibration data. Financial asset returns violate exchangeability through volatility clustering, leverage effects, and macro shifts.
   - While ACI guarantees asymptotic validity $\frac{1}{T}\sum_{t=1}^T \text{err}_t \to \alpha$, finite-sample conditional coverage $\mathbb{P}(r_t < \text{VaR}_t \mid \mathcal{F}_{t-1})$ may fluctuate locally during sudden market shocks.
2. **Asymptotic Marginal vs. Conditional Coverage**:
   - Standard split conformal prediction delivers marginal coverage over randomly drawn days. A desk requires conditional coverage for *tomorrow's close*. Sliding CQR and ACI mitigate this via localized sliding buffers, but do not eliminate conditional auto-correlation of errors during abrupt crashes.
3. **Survivorship Bias Mitigation**:
   - Benchmarking exclusively on surviving liquid blue-chips artificially deflates tail risk. This repository incorporates non-survivors and stressed historical constituents (`YESBANK.NS`, `IDEA.NS`, `ZEEL.NS`) to prevent survivorship bias from invalidating left-tail risk models.
4. **Expected Shortfall Approximation**:
   - Conformal Prediction is fundamentally an empirical quantile estimator; it does not yield probability density shapes beyond the estimated quantile. $\text{ES}_\alpha$ cannot be directly conformalized without distributional assumptions. This library approximates ES through numerical integration of ACI calibrated quantiles.

---

## 6. System Architecture & Production Packaging

```
conformal-var-india/
├── src/conformal_var/
│   ├── config.py         # Strict Pydantic v2 configuration schema
│   ├── data/loader.py    # Survivorship-aware NSE data pipeline (Parquet cache)
│   ├── features/         # Causal volatility & technical features (.shift(1))
│   ├── models/           # Historical Simulation, RiskMetrics, GARCH, LightGBM
│   ├── conformal/        # Split, Sliding CQR, Online ACI, Vol-Scaled ACI
│   ├── backtest/         # Kupiec POF, Christoffersen, Basel Traffic Light
│   ├── portfolio/        # Multi-asset return aggregation, Conformal ES
│   └── api/              # Production FastAPI service with Redis/LRU cache
├── artifacts/            # Parquet tables, diagnostic plots, scorecards
├── scripts/              # Ingestion, research experiment, plot generators
├── tests/                # 90 pytest analytical & integration tests
├── Dockerfile            # Multi-stage production container
└── docker-compose.yml    # FastAPI + Redis 7 stack
```

---

## 7. Quickstart & Verification

### Local Setup & Testing
```bash
# Clone and install dependencies
git clone https://github.com/RealDarthVader/conformal-var-india.git
cd conformal-var-india
pip install -r requirements.txt

# Run the complete test suite (90 passing tests)
pytest

# Regenerate diagnostic plots
python scripts/generate_plots.py
```

### Docker Compose Deployment
```bash
# Spin up FastAPI application and Redis cache
docker compose up --build -d

# Verify health status
curl -s http://localhost:8000/health | jq .
```

### API Inference Example: Portfolio VaR & Expected Shortfall
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

#### Example JSON Response
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
