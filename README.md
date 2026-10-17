# conformal-var-india

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/mypy-strict-blue)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Uncertainty-Calibrated Value-at-Risk (VaR) Engine for Indian Equities**

`conformal-var-india` provides finite-sample coverage guarantees for downside market risk forecasting on National Stock Exchange (NSE) indices and equities using Conformal Prediction, volatility forecasting (GARCH, Quantile Regressors, LightGBM), and statistical backtesting suites (Kupiec POF, Christoffersen conditional coverage).

---

## 🏛️ Architecture & Directory Layout

```
conformal-var-india/
├── src/
│   └── conformal_var/
│       ├── __init__.py
│       ├── config.py         # Strict Pydantic v2 configuration schema
│       ├── data/             # Market data ingestion (NSE/Yahoo Finance) & storage
│       ├── features/         # Return & volatility feature transformations
│       ├── models/           # GARCH, LightGBM, and baseline parametric models
│       ├── conformal/        # Nonconformity scores, calibration, and conformal bounds
│       ├── backtest/         # Kupiec POF, Christoffersen, traffic light tests
│       ├── portfolio/        # Multi-asset portfolio risk aggregation & Euler allocation
│       └── api/              # FastAPI real-time risk inference endpoints
├── tests/                    # Unit, validation, and statistical sanity tests
├── scripts/                  # CLI runners for data ingestion, benchmarking, and plots
├── artifacts/                # Parquet datasets, diagnostic plots, markdown benchmark tables
│   ├── data/
│   ├── plots/
│   └── tables/
├── pyproject.toml            # Build system, project metadata, tool configurations
├── requirements.txt          # Pinned dependency requirements
├── config.yaml               # Canonical YAML risk engine settings
└── README.md
```

---

## ⚙️ Configuration & Risk Parameters

Configuration is loaded from `config.yaml` with strict validation via Pydantic:

- **Target Quantiles**: $\alpha \in \{0.05, 0.01\}$ corresponding to 95% and 99% confidence levels.
- **Rolling Windows**:
  - Training Window: **756 trading days** (~3 years).
  - Calibration Window: **252 trading days** (1 year).
- **Indian Market Regimes**:
  - `calm_bull`: 2017-01-01 to 2018-01-31
  - `ilfs_liquidity_crisis`: 2018-09-01 to 2019-03-31
  - `covid_crash`: 2020-02-01 to 2020-06-30
  - `rate_hike_tightening`: 2022-01-01 to 2022-12-31

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/RealDarthVader/conformal-var-india.git
cd conformal-var-india
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Verification & Testing

Run the test suite and type check:

```bash
pytest tests/test_config.py
mypy src/
ruff check src/ tests/
```
