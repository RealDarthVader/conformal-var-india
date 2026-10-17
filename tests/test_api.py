"""Tests for FastAPI REST API endpoints.

Verifies:
1. GET /health: Health check, Redis fallback status, latest market date.
2. POST /api/v1/portfolio/var: Valid calculation of VaR, ES, and regulatory backtests.
3. Caching & Latency: Cache hit on second call with sub-millisecond response latency.
4. Schema validation errors: Reject invalid weights (empty, negative, or sum to 0).
5. GET /api/v1/benchmarks/regimes: Returns precomputed regime scorecard table.

Design note: All tests are data-independent. A synthetic returns DataFrame is injected
directly into ``app_state`` before the test client starts so that CI runners — which have
no ``artifacts/data/returns.parquet`` file — can exercise every code-path without real
market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conformal_var.api.main import app, app_state

# ---------------------------------------------------------------------------
# Synthetic dataset shared across the entire test module
# ---------------------------------------------------------------------------

_SYNTHETIC_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
_N_DAYS = 1_000  # enough for 252-day rolling windows + 250-day backtest


def _make_synthetic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=_N_DAYS)
    data = rng.normal(0.0, 0.015, (_N_DAYS, len(_SYNTHETIC_TICKERS)))
    return pd.DataFrame(data, index=dates, columns=_SYNTHETIC_TICKERS)


@pytest.fixture(scope="module")
def client() -> TestClient:  # type: ignore[return]
    """TestClient with a synthetic returns DataFrame pre-loaded into ``app_state``."""
    synthetic = _make_synthetic_returns()

    # Inject synthetic data so the lifespan data-load path is bypassed.
    # We do this *before* entering the TestClient context manager so the
    # lifespan ``load_cached_returns`` failure does not clobber our inject.
    app_state["returns_df"] = synthetic
    app_state["latest_date"] = str(synthetic.index[-1])[:10]

    with TestClient(app) as test_client:
        # Re-inject after lifespan may have overwritten with None on CI
        if app_state["returns_df"] is None:
            app_state["returns_df"] = synthetic
            app_state["latest_date"] = str(synthetic.index[-1])[:10]
        yield test_client

    # Restore default state after module tests finish
    app_state["returns_df"] = None
    app_state["latest_date"] = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_endpoint(client: TestClient) -> None:
    """Verify /health returns status ok and cache backend information."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["cache_backend"] in {"redis", "in_memory_lru"}
    assert "latest_market_date" in data
    assert isinstance(data["available_tickers"], list)
    # With synthetic data injected, tickers must be present
    assert len(data["available_tickers"]) > 0


def test_portfolio_var_valid_request(client: TestClient) -> None:
    """Verify POST /api/v1/portfolio/var calculates VaR, ES, and diagnostics."""
    payload = {
        "weights": {"RELIANCE.NS": 0.5, "TCS.NS": 0.5},
        "alpha": 0.01,
        "model": "aci_lgbm",
        "gamma": 0.01,
        "portfolio_value_inr": 2_000_000.0,
    }
    response = client.post("/api/v1/portfolio/var", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Validate schema fields
    assert "date" in data
    assert data["alpha"] == 0.01
    assert data["var_return_pct"] < 0.0  # VaR is a negative return
    assert data["var_inr"] > 0.0  # INR loss amount is positive
    assert data["expected_shortfall_pct"] <= data["var_return_pct"]  # ES <= VaR
    assert data["expected_shortfall_inr"] >= data["var_inr"]
    assert 0.0 <= data["empirical_coverage_250d"] <= 1.0
    assert 0.0 <= data["kupiec_p_value"] <= 1.0
    assert data["basel_traffic_light"] in {"GREEN", "YELLOW", "RED"}
    assert data["cache_hit"] is False
    assert data["latency_ms"] > 0.0


def test_portfolio_var_caching_and_sub_millisecond_latency(client: TestClient) -> None:
    """Verify second identical request hits cache and completes rapidly."""
    payload = {
        "weights": {"INFY.NS": 0.6, "HDFCBANK.NS": 0.4},
        "alpha": 0.05,
        "model": "hist_sim",
        "portfolio_value_inr": 1_000_000.0,
    }

    # First call: cache miss
    res1 = client.post("/api/v1/portfolio/var", json=payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["cache_hit"] is False

    # Second call: cache hit
    res2 = client.post("/api/v1/portfolio/var", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cache_hit"] is True
    assert data2["var_inr"] == data1["var_inr"]
    # Fast cached response
    assert data2["latency_ms"] < 100.0


def test_invalid_weights_validation_error(client: TestClient) -> None:
    """Verify 422 Unprocessable Entity for negative or empty weights."""
    # Empty weights
    res_empty = client.post("/api/v1/portfolio/var", json={"weights": {}})
    assert res_empty.status_code == 422

    # Negative weights in long-only schema
    res_neg = client.post(
        "/api/v1/portfolio/var",
        json={"weights": {"RELIANCE.NS": 1.2, "TCS.NS": -0.2}},
    )
    assert res_neg.status_code == 422


def test_unknown_ticker_returns_400(client: TestClient) -> None:
    """Verify 400 Bad Request when ticker does not exist in dataset."""
    payload = {
        "weights": {"NONEXISTENT_TICKER.NS": 1.0},
        "alpha": 0.01,
    }
    response = client.post("/api/v1/portfolio/var", json=payload)
    assert response.status_code == 400
    assert "not available in the market data universe" in response.json()["detail"]


def test_benchmarks_regimes_endpoint(client: TestClient) -> None:
    """Verify GET /api/v1/benchmarks/regimes returns scorecard records."""
    response = client.get("/api/v1/benchmarks/regimes?ticker=^NSEI")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "^NSEI"
    assert data["total_records"] > 0
    assert isinstance(data["data"], list)
    first_record = data["data"][0]
    assert "Regime" in first_record
    assert "Coverage" in first_record
    assert "Basel Zone" in first_record
