# conformal-var-india

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- Add a "Live demo" badge here once the Streamlit app is deployed -->

Value-at-Risk (VaR) and Expected Shortfall (ES) for Indian equities, built by wrapping LightGBM quantile regression in conformal calibration (sliding-window CQR and Adaptive Conformal Inference), then backtested against classical baselines with the tests risk desks actually use (Kupiec, Christoffersen, Basel traffic lights).

**TL;DR**

- Conformal calibration fixes the under-coverage of a raw LightGBM quantile model. Online ACI kept violations near nominal through the YES Bank collapse (6 violations vs. 56 uncalibrated).
- It does **not** beat a well-tuned GARCH(1,1)-Student-t on the full Nifty 50 sample. GARCH-t has the lowest pinball loss and the narrowest bands; the conformal variants buy better coverage with roughly 35-40% wider bands at the 1% level.
- In the YES Bank crash, GARCH-t was not statistically rejected (12 violations, Kupiec p = 0.115). ACI had fewer violations (6) at about 67% wider bands.

---

## Contents

1. [Findings](#1-findings)
2. [Method](#2-method)
3. [Evaluation protocol](#3-evaluation-protocol)
4. [Results](#4-results)
5. [Limitations](#5-limitations)
6. [Future work](#6-future-work)
7. [Architecture](#7-architecture)
8. [Quickstart](#8-quickstart)

---

## 1. Findings

**Research question.** Does Adaptive Conformal Inference wrapped around a machine-learning quantile regressor beat a well-tuned GARCH(1,1)-Student-t or a volatility-scaled residual band on Indian equity tails?

**Data.** 3,356 out-of-sample trading days (about 13 years) on the Nifty 50 index (`^NSEI`), plus a single-stock stress test on `YESBANK.NS` (2018-2020, 736 days).

**Answer: not on the full sample, but conformal calibration solves a real problem.**

1. **Conformal layers repair a miscalibrated ML model.** Uncalibrated LightGBM under-covers at both levels (1% target: 97.81% coverage, Kupiec p = 0.0000). Sliding CQR brings it to 98.88% (p = 0.4846), and the ACI variants bring it to 98.60-98.72%.
2. **They do not beat GARCH-t on a proper scoring rule.** At the 1% level GARCH-t has the lowest pinball loss of every model (0.000364 vs. 0.000413-0.000436 for the conformal variants) and the narrowest bands among the models with coverage above 98.5% (0.0231 vs. 0.0311-0.0320). At the 5% level the GARCH and EWMA models also have lower pinball loss than every LightGBM variant. GARCH-t is marginally rejected by Kupiec at 1% (p = 0.0452); so is LightGBM + Online ACI (p = 0.0308).
3. **During the YES Bank collapse, ACI has fewer violations but GARCH-t is not rejected.** Over 736 days at the 1% level, about 7.4 violations are expected. Uncalibrated LightGBM had 56 (Basel Red). GARCH-t had 12 (Kupiec p = 0.1152, Basel Green, mean width 0.1308). Online ACI had 6 (p = 0.6027, mean width 0.2180, about 67% wider than GARCH-t). The sample includes single-day log-return shocks of -0.8235 (-56.1% simple return).
4. **Volatility scaling did not help in the crash.** Vol-Scaled ACI matched Sliding CQR's violation count (7) at about twice the width (0.4477 vs. 0.2114). The likely cause is that dividing residuals by a volatility estimate that spikes after extreme shocks over-inflates the bands. This is a hypothesis; no ablation here isolates it.

**What this does not show.** The index results are one series, and the crash result is one stock chosen with hindsight. See [Limitations](#5-limitations) and [Future work](#6-future-work), in particular the missing GARCH-t + ACI ablation, which is the cleanest test of whether the ML base model adds anything beyond the conformal layer.

---

## 2. Method

Let `r_t = ln(P_t / P_{t-1})` be the daily log return, and let `X_t` be features observable before day `t` (all features are lagged with `.shift(1)`).

**Quantile loss.** For miscoverage level α and predicted quantile `v`:

$$\mathcal{L}_\alpha(r_t, v) = \left(\alpha - \mathbb{I}\{r_t < v\}\right)(r_t - v)$$

**Conformity score.** For a lower-tail quantile forecast `q_t` (the LightGBM quantile at level α), a violation `r_t < q_t` gives a positive score:

$$s_t = q_t - r_t$$

**Sliding-window CQR.** With a rolling window of the last `M` scores:

$$\widehat{\text{VaR}}_t = q_t - \hat{Q}_{1-\alpha}\left(\{s_{t-M}, \dots, s_{t-1}\}\right)$$

where `Q̂` is the empirical quantile at level `⌈(M+1)(1-α)⌉ / M`.

**Online ACI (Gibbs & Candès, 2021).** Track an adaptive miscoverage level `α_t` with step size γ:

$$\widehat{\text{VaR}}_t = q_t - \hat{Q}_{1-\alpha_t}(s), \qquad \text{err}_t = \mathbb{I}\{r_t < \widehat{\text{VaR}}_t\}$$

$$\alpha_{t+1} = \text{clip}\left(\alpha_t + \gamma(\alpha - \text{err}_t),\ \alpha_{\min},\ \alpha_{\max}\right)$$

A violation lowers `α_t`, which widens the next band; a run of non-violations raises it.

**Volatility-scaled ACI.** Scores are normalized by a causal volatility estimate `σ_t`:

$$s_t^{\text{scaled}} = \frac{q_t - r_t}{\sigma_t}, \qquad \widehat{\text{VaR}}_t = q_t - \sigma_t \cdot \hat{Q}_{1-\alpha_t}(s^{\text{scaled}})$$

**Expected Shortfall (approximation).** Conformal methods calibrate quantiles, not tail expectations, so ES is approximated by averaging ACI quantiles over a 10-point grid `τ_j = jα/10`:

$$\widehat{\text{ES}}_t(\alpha) \approx \frac{1}{10}\sum_{j=1}^{10} \widehat{\text{VaR}}_t(\tau_j)$$

By construction `ES ≤ VaR` in return space. See the ES limitation below.

**Backtests.**

- **Kupiec POF** tests unconditional coverage (violation rate equals α).
- **Christoffersen** tests independence of violations (no clustering).
- **Basel traffic lights** are defined for 99% VaR over 250 trading days: Green 0-4 violations, Yellow 5-9, Red 10 or more. They are not applied at α = 0.05.

---

## 3. Evaluation protocol

All results are out-of-sample. The items below define how the numbers were produced.

- **Data source and snapshot:** [TODO: source, adjusted vs. raw prices, snapshot date]
- **Training and retraining:** [TODO: walk-forward scheme and LightGBM retraining frequency]
- **Hyperparameter selection:** [TODO: how γ, `M`, `α_min`, `α_max` and the LightGBM parameters were chosen, and on which period. Confirm none were tuned on the test period]
- **Basel zone assignment:** [TODO: how zones are assigned for samples longer than 250 days, e.g. violations scaled to a 250-day equivalent or worst rolling window]

---

## 4. Results

### Table 1: Nifty 50 (`^NSEI`), 3,356 out-of-sample days

Bold marks the best value in each block for coverage error, pinball loss (lower is better), and Kupiec p-value.

| Model | Target α | Coverage | Coverage error | Mean width | Pinball loss | Kupiec p | Basel zone † |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Historical Simulation (252d) | 0.05 | 94.32% | 0.68% | 0.0148 | 0.001224 | 0.0800 | N/A |
| RiskMetrics EWMA (λ=0.94) | 0.05 | 94.20% | 0.80% | 0.0151 | **0.001138** | 0.0397 | N/A |
| GARCH(1,1)-Normal | 0.05 | 94.84% | 0.16% | 0.0154 | **0.001138** | 0.6704 | N/A |
| GARCH(1,1)-Student-t | 0.05 | 94.59% | 0.41% | 0.0148 | 0.001139 | 0.2918 | N/A |
| Uncalibrated LightGBM | 0.05 | 93.20% | 1.80% | 0.0137 | 0.001238 | 0.0000 | N/A |
| LightGBM + Online ACI | 0.05 | 94.90% | 0.10% | 0.0178 | 0.001218 | 0.7895 | N/A |
| LightGBM + Vol-Scaled ACI | 0.05 | 94.93% | **0.07%** | 0.0188 | 0.001248 | **0.8513** | N/A |
| Historical Simulation (252d) | 0.01 | 98.24% | 0.76% | 0.0258 | 0.000444 | 0.0001 | Green |
| RiskMetrics EWMA (λ=0.94) | 0.01 | 98.03% | 0.97% | 0.0214 | 0.000372 | 0.0000 | Yellow |
| GARCH(1,1)-Normal | 0.01 | 98.30% | 0.70% | 0.0217 | 0.000368 | 0.0002 | Green |
| GARCH(1,1)-Student-t | 0.01 | 98.63% | 0.37% | 0.0231 | **0.000364** | 0.0452 | Green |
| Uncalibrated LightGBM | 0.01 | 97.81% | 1.19% | 0.0222 | 0.000480 | 0.0000 | Yellow |
| LightGBM + Sliding CQR | 0.01 | 98.88% | **0.12%** | 0.0315 | 0.000435 | **0.4846** | Green |
| LightGBM + Online ACI | 0.01 | 98.60% | 0.40% | 0.0320 | 0.000436 | 0.0308 | Green |
| LightGBM + Vol-Scaled ACI | 0.01 | 98.72% | 0.28% | 0.0311 | 0.000413 | 0.1276 | Green |

Notes:

- A Kupiec p-value below 0.05 rejects correct unconditional coverage. At the 1% level, GARCH-t (0.0452) and LightGBM + Online ACI (0.0308) are rejected at the 5% significance level, along with several other models.
- "Mean width" is the average VaR magnitude; narrower is better only at comparable coverage.
- † Basel zones apply to 99% VaR only. See [Evaluation protocol](#3-evaluation-protocol) for how zones are assigned over the full sample.

### Table 2: `YESBANK.NS` left-tail collapse (2018-2020, α = 0.01, 736 days)

Expected violations at the nominal rate: about 7.4.

| Model | Coverage | Violations | Mean width | Kupiec p | Christoffersen p | Basel zone † |
| --- | --- | --- | --- | --- | --- | --- |
| Uncalibrated LightGBM | 92.39% | 56 | 0.0732 | 0.0000 | 0.0000 | Red |
| RiskMetrics EWMA (λ=0.94) | 97.55% | 18 | 0.1183 | 0.0009 | 0.4557 | Yellow |
| Historical Simulation (252d) | 97.96% | 15 | 0.1240 | 0.0131 | 0.0333 | Yellow |
| GARCH(1,1)-Student-t | 98.37% | 12 | 0.1308 | 0.1152 | 0.1840 | Green |
| LightGBM + Sliding CQR | 99.05% | 7 | 0.2114 | 0.8930 | 0.7137 | Green |
| LightGBM + Online ACI (γ=0.01) | 99.18% | 6 | 0.2180 | 0.6027 | 0.7533 | Green |
| LightGBM + Vol-Scaled ACI | 99.05% | 7 | 0.4477 | 0.8930 | 0.7137 | Green |

The conformal variants land closest to the expected violation count, and GARCH-t is not rejected, so the real difference between them is band width: the conformal bands are 62-71% wider than GARCH-t's, and Vol-Scaled ACI about 3.4 times wider.

Diagnostic plots (crash replay with the `α_t` trajectory, coverage vs. width) are generated by `scripts/generate_plots.py` into `artifacts/plots/` and shown interactively in the Streamlit demo. Per-regime scorecards are in `artifacts/`.

---

## 5. Limitations

1. **Exchangeability.** Classical conformal guarantees assume exchangeable data. Returns violate this through volatility clustering and regime shifts. ACI relaxes the assumption but only guarantees long-run average coverage.
2. **The ACI guarantee is loose at this sample size.** The worst-case bound on the gap between average violation rate and α is `(max(α₁, 1-α₁) + γ) / (γT)`. With γ = 0.01 and T = 3,356 this is about 3 percentage points, which is larger than the 1% target. The empirical results, not the theorem, carry the evidence. Clipping `α_t` can also weaken the guarantee.
3. **Marginal vs. conditional coverage.** Average coverage over time does not imply correct coverage conditional on today's market state. Violation clustering is tested with Christoffersen, which is reported for the stress test but not yet for the index table.
4. **Limited breadth and hindsight selection.** The main table is one index. The stress test is one stock chosen after the fact because it collapsed. `YESBANK.NS`, `IDEA.NS` and `ZEEL.NS` are stressed names, not point-in-time index constituents, so this does not remove survivorship bias. Results across a broad cross-section of stocks are not yet reported.
5. **Weak ML base model.** Uncalibrated LightGBM has worse pinball loss than GARCH-t (0.000480 vs. 0.000364 at 1%). The conformal layer fixes calibration, not the quality of the underlying quantile forecast.
6. **Expected Shortfall.** The 10-point grid uses right-endpoint levels `jα/10` and excludes the extreme tail, which biases ES toward milder values; a midpoint or finer grid would reduce this. ES is not backtested (for example with a joint VaR-ES score).

---

## 6. Future work

- **GARCH-t + ACI:** conformalize the GARCH-t quantile instead of LightGBM. This directly tests whether the ML model adds anything beyond the conformal layer.
- **Cross-sectional evaluation:** run all models on every Nifty 50 stock using point-in-time constituents, and report the fraction passing Kupiec and Christoffersen.
- **Christoffersen on the index table** and a conditional coverage joint test.
- **ES backtesting** and a midpoint or finer ES grid.
- **Sensitivity** to γ, `M` and `α_min`/`α_max`.

---

## 7. Architecture

```text
conformal-var-india/
├── src/conformal_var/
│   ├── config.py         # Pydantic v2 configuration schema
│   ├── data/loader.py    # NSE data pipeline (Parquet cache)
│   ├── features/         # Causal lagged volatility and tail features (.shift(1))
│   ├── models/           # Historical Simulation, RiskMetrics, GARCH, LightGBM
│   ├── conformal/        # Split, Sliding CQR, Online ACI, Vol-Scaled ACI
│   ├── backtest/         # Kupiec POF, Christoffersen, Basel traffic light
│   ├── portfolio/        # Multi-asset return aggregation, ES grid
│   └── api/              # FastAPI service with Redis/LRU cache
├── streamlit_app.py      # Three-tab Streamlit demo (precomputed, no model fitting)
├── artifacts/            # Parquet datasets, plots, regime scorecards
├── scripts/              # Ingestion, demo precompute, experiments, plots
├── tests/                # 90 pytest tests: analytical, leakage, integration
├── Dockerfile            # Multi-stage container (API and demo targets)
└── docker-compose.yml    # FastAPI + Redis 7 + Streamlit
```

---

## 8. Quickstart

### Install and test

```bash
git clone https://github.com/RealDarthVader/conformal-var-india.git
cd conformal-var-india
pip install -r requirements.txt

pytest                                      # 90 tests

python scripts/run_research_experiment.py   # regenerate result tables
python scripts/generate_plots.py            # regenerate plots
```

### Streamlit demo

```bash
python scripts/precompute_demo.py   # one-off, about 2 minutes
streamlit run streamlit_app.py
```

Open http://localhost:8501 for the **Crash Replay**, **Model Comparison**, and **Portfolio VaR & ES** tabs.

### Docker

```bash
docker compose up --build -d
curl -s http://localhost:8000/health | jq .
```

- Streamlit demo: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs

### API example: portfolio VaR and ES

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

Example response:

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

---

## License

MIT
