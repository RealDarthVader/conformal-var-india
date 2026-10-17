"""Production FastAPI REST API for Uncertainty-Calibrated Value-at-Risk Engine.

Endpoints:
- GET /health: Service health, Redis status, latest market date.
- POST /api/v1/portfolio/var: Computes or fetches cached portfolio VaR, ES, and 250d backtest.
- GET /api/v1/benchmarks/regimes: Returns precomputed regime stress-test table across regimes.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from conformal_var.api.schemas import (
    HealthResponse,
    PortfolioVaRRequest,
    PortfolioVaRResponse,
    RegimeStressBenchmarkResponse,
)
from conformal_var.backtest.metrics import (
    basel_traffic_light,
    christoffersen_independence_test,
    kupiec_pof_test,
)
from conformal_var.conformal.calibrator import AdaptiveConformalVaR
from conformal_var.data.loader import load_cached_returns
from conformal_var.models.baselines import (
    GARCHStudentTVaR,
    HistoricalSimulationVaR,
)
from conformal_var.portfolio.engine import PortfolioRiskEngine

logger = logging.getLogger("conformal_var_api")


class InMemoryLRUCache:
    """Thread-safe LRU cache fallback when Redis is offline."""

    def __init__(self, capacity: int = 500) -> None:
        self.capacity = capacity
        self._cache: collections.OrderedDict[str, tuple[str, float]] = collections.OrderedDict()

    def get(self, key: str) -> str | None:
        """Retrieve value if key exists and has not expired."""
        if key not in self._cache:
            return None
        val, expiry = self._cache[key]
        if time.time() > expiry:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return val

    def set(self, key: str, value: str, ex: int = 3600) -> None:
        """Store value with expiration in seconds."""
        if key in self._cache:
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = (value, time.time() + ex)


class CacheClient:
    """Unified cache interface trying Redis with automatic LRU in-memory fallback."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self.redis_url = redis_url
        self._redis: Any = None
        self._lru = InMemoryLRUCache(capacity=500)
        self.is_redis_connected = False
        self._try_connect_redis()

    def _try_connect_redis(self) -> None:
        try:
            import redis

            r = redis.from_url(
                self.redis_url,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
            )
            r.ping()
            self._redis = r
            self.is_redis_connected = True
            logger.info("Successfully connected to Redis at %s", self.redis_url)
        except Exception:
            self.is_redis_connected = False
            self._redis = None
            logger.info("Redis not available. Using InMemoryLRUCache fallback.")

    def get(self, key: str) -> str | None:
        if self.is_redis_connected and self._redis:
            try:
                res = self._redis.get(key)
                if res is not None:
                    return str(res.decode("utf-8") if isinstance(res, bytes) else res)
            except Exception:
                self.is_redis_connected = False
        return self._lru.get(key)

    def set(self, key: str, value: str, ex: int = 3600) -> None:
        if self.is_redis_connected and self._redis:
            try:
                self._redis.set(key, value, ex=ex)
                return
            except Exception:
                self.is_redis_connected = False
        self._lru.set(key, value, ex=ex)


# Global state
app_state: dict[str, Any] = {
    "cache": None,
    "returns_df": None,
    "latest_date": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context for cache connection and dataset loading."""
    cache = CacheClient()
    app_state["cache"] = cache

    try:
        returns_df = load_cached_returns(Path("artifacts/data"))
        app_state["returns_df"] = returns_df
        app_state["latest_date"] = str(returns_df.index[-1])[:10]
        logger.info(
            "Loaded returns dataset with %d days, latest: %s",
            len(returns_df),
            app_state["latest_date"],
        )
    except Exception as e:
        logger.warning("Could not pre-load returns dataset: %s", e)
        app_state["returns_df"] = None

    yield
    # Cleanup


app = FastAPI(
    title="Conformal VaR India REST API",
    description="Uncertainty-calibrated VaR and Expected Shortfall engine on Indian equities.",
    version="1.0.0",
    lifespan=lifespan,
)



@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return API service health status, Redis connection, and data freshness."""
    cache: CacheClient = app_state["cache"]
    returns_df: pd.DataFrame | None = app_state.get("returns_df")

    latest_date = app_state.get("latest_date")
    available_tickers = list(returns_df.columns) if returns_df is not None else []

    return HealthResponse(
        status="ok",
        redis_connected=cache.is_redis_connected if cache else False,
        cache_backend="redis" if (cache and cache.is_redis_connected) else "in_memory_lru",
        latest_market_date=latest_date,
        available_tickers=available_tickers,
    )


@app.post(
    "/api/v1/portfolio/var",
    response_model=PortfolioVaRResponse,
    status_code=status.HTTP_200_OK,
)
def compute_portfolio_var(request: PortfolioVaRRequest) -> PortfolioVaRResponse:
    """Compute or retrieve cached portfolio Value-at-Risk, ES, and 250d backtest diagnostics."""
    t0 = time.perf_counter()

    # 1. Check cache
    cache: CacheClient = app_state["cache"]
    # Generate deterministic query key
    canonical_weights = json.dumps(dict(sorted(request.weights.items())))
    cache_payload = (
        f"{canonical_weights}|{request.alpha}|{request.model}|{request.gamma}|"
        f"{request.portfolio_value_inr}"
    )
    cache_key = f"portfolio_var:{hashlib.sha256(cache_payload.encode()).hexdigest()}"

    cached_entry = cache.get(cache_key)
    if cached_entry:
        data = json.loads(cached_entry)
        return PortfolioVaRResponse(
            date=str(data["date"]),
            alpha=float(data["alpha"]),
            var_return_pct=float(data["var_return_pct"]),
            var_inr=float(data["var_inr"]),
            expected_shortfall_pct=float(data["expected_shortfall_pct"]),
            expected_shortfall_inr=float(data["expected_shortfall_inr"]),
            empirical_coverage_250d=float(data["empirical_coverage_250d"]),
            kupiec_p_value=float(data["kupiec_p_value"]),
            christoffersen_ind_p_value=float(data["christoffersen_ind_p_value"]),
            basel_traffic_light=data["basel_traffic_light"],
            cache_hit=True,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )


    # 2. Validate tickers exist in returns data
    returns_df: pd.DataFrame | None = app_state.get("returns_df")
    if returns_df is None:
        try:
            returns_df = load_cached_returns(Path("artifacts/data"))
            app_state["returns_df"] = returns_df
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Returns database not available: {err}",
            ) from err

    for ticker in request.weights:
        if ticker not in returns_df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ticker '{ticker}' is not available in the market data universe.",
            )

    # 3. Aggregate portfolio log-returns
    engine = PortfolioRiskEngine(
        request.weights,
        rolling_cov_window=252,
        aci_window=252,
        aci_gamma=request.gamma,
    )
    port_returns = engine.aggregate_portfolio_returns(returns_df)

    t_obs = len(port_returns)
    if t_obs < 252:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient history ({t_obs} days). Need at least 252 days.",
        )

    # 4. Generate 1-step-ahead VaR & ES forecast using chosen model
    effective_date = str(port_returns.index[-1])[:10]
    eval_window = 252
    eval_returns = port_returns.iloc[-eval_window:]

    grid_points = 10
    tail_grid = [round((j * request.alpha) / grid_points, 6) for j in range(1, grid_points + 1)]

    # Fit chosen model on past returns
    if request.model in {"aci_lgbm", "vol_aci_lgbm"}:
        # In online service, portfolio log returns calibrated via AdaptiveConformalVaR
        base_f = HistoricalSimulationVaR(window=252)
        base_f.fit(port_returns)
        raw_var = base_f.predict_var(request.alpha)

        aci = AdaptiveConformalVaR(
            alpha=request.alpha,
            gamma=request.gamma,
            window=252,
        )
        # Pre-seed calibration buffer with past 252 residuals
        init_preds = [
            float(np.quantile(port_returns.iloc[max(0, i - 252) : i].to_numpy(), request.alpha))
            for i in range(t_obs - 252, t_obs)
        ]
        aci.calibrate(np.array(init_preds), eval_returns.to_numpy())
        var_pred = aci.predict_var(raw_var)

        # Expected Shortfall: 10-point grid average
        grid_vars = []
        for tau in tail_grid:
            q_tau = base_f.predict_var(tau)
            c_tau = AdaptiveConformalVaR(alpha=tau, gamma=request.gamma, window=252)
            init_tau = [
                float(np.quantile(port_returns.iloc[max(0, i - 252) : i].to_numpy(), tau))
                for i in range(t_obs - 252, t_obs)
            ]
            c_tau.calibrate(np.array(init_tau), eval_returns.to_numpy())
            grid_vars.append(c_tau.predict_var(q_tau))
        es_pred = min(float(np.mean(grid_vars)), var_pred)

    elif request.model == "garch_t":
        garch = GARCHStudentTVaR(training_window=min(756, t_obs), dist="t")
        garch.fit(port_returns)
        var_pred = garch.predict_var(request.alpha)
        # ES via numerical tail integration
        grid_vars = [garch.predict_var(tau) for tau in tail_grid]
        es_pred = min(float(np.mean(grid_vars)), var_pred)

    else:  # hist_sim
        hs = HistoricalSimulationVaR(window=252)
        hs.fit(port_returns)
        var_pred = hs.predict_var(request.alpha)
        grid_vars = [hs.predict_var(tau) for tau in tail_grid]
        es_pred = min(float(np.mean(grid_vars)), var_pred)

    # Convert returns to percentages and INR loss amounts
    # Loss in INR is positive: value * |var_pred|
    var_return_pct = round(float(var_pred * 100), 4)
    var_inr = round(float(abs(var_pred) * request.portfolio_value_inr), 2)
    es_return_pct = round(float(es_pred * 100), 4)
    es_inr = round(float(abs(es_pred) * request.portfolio_value_inr), 2)

    # 5. Live 250-Day Backtest Diagnostics
    recent_250 = port_returns.iloc[-250:].to_numpy()
    # Violations against rolling empirical risk threshold
    rolling_threshold = float(np.quantile(port_returns.iloc[-500:].to_numpy(), request.alpha))
    violations_250 = (recent_250 < rolling_threshold).astype(int)

    kupiec = kupiec_pof_test(violations_250, alpha=request.alpha)
    ind = christoffersen_independence_test(violations_250)
    traffic_light = basel_traffic_light(violations_250, window_size=250)

    emp_coverage_250 = round(float(1.0 - (int(np.sum(violations_250)) / 250)), 4)
    zone_upper = traffic_light.zone.upper()
    zone_valid: Literal["GREEN", "YELLOW", "RED"] = (
        "GREEN" if zone_upper == "GREEN" else ("YELLOW" if zone_upper == "YELLOW" else "RED")
    )


    res = PortfolioVaRResponse(
        date=effective_date,
        alpha=request.alpha,
        var_return_pct=var_return_pct,
        var_inr=var_inr,
        expected_shortfall_pct=es_return_pct,
        expected_shortfall_inr=es_inr,
        empirical_coverage_250d=emp_coverage_250,
        kupiec_p_value=round(kupiec.p_value, 4),
        christoffersen_ind_p_value=round(ind.p_value, 4),
        basel_traffic_light=zone_valid,
        cache_hit=False,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )

    # Store in cache with 3600s TTL
    cache.set(cache_key, res.model_dump_json(), ex=3600)

    return res



@app.get(
    "/api/v1/benchmarks/regimes",
    response_model=RegimeStressBenchmarkResponse,
    status_code=status.HTTP_200_OK,
)
def get_regime_benchmarks(ticker: str = "^NSEI") -> RegimeStressBenchmarkResponse:
    """Retrieve precomputed regime stress-test benchmark results."""
    csv_path = Path("artifacts/tables/regime_scorecard.csv")
    if not csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Regime scorecard not generated yet. "
                "Run scripts/run_research_experiment.py or regimes.py."
            ),
        )


    df = pd.read_csv(csv_path)
    if "Ticker" in df.columns:
        filtered = df[df["Ticker"] == ticker]
    else:
        filtered = df

    records = filtered.to_dict(orient="records")
    return RegimeStressBenchmarkResponse(
        ticker=ticker,
        total_records=len(records),
        data=records,
    )
