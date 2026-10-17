"""Strict Pydantic v2 schemas for the Conformal VaR REST API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PortfolioVaRRequest(BaseModel):
    """Request model for computing portfolio Value-at-Risk and Expected Shortfall."""

    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float] = Field(
        ...,
        description="Dictionary mapping ticker symbols to portfolio weights.",
        examples=[{"RELIANCE.NS": 0.5, "TCS.NS": 0.5}],
    )
    alpha: float = Field(
        default=0.01,
        description="Target risk quantile (0.01 for 99% VaR, 0.05 for 95% VaR).",
    )

    model: Literal["aci_lgbm", "vol_aci_lgbm", "garch_t", "hist_sim"] = Field(
        default="aci_lgbm",
        description="VaR forecasting model architecture.",
    )
    gamma: float = Field(
        default=0.01,
        description="ACI step adaptation learning rate.",
        gt=0.0,
        le=0.1,
    )
    portfolio_value_inr: float = Field(
        default=1_000_000.0,
        description="Current portfolio market value in Indian Rupees (INR).",
        gt=0.0,
    )

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, alpha: float) -> float:
        """Validate target risk quantile is strictly 0.01 or 0.05."""
        if alpha not in {0.01, 0.05}:
            raise ValueError(f"alpha must be either 0.01 or 0.05, got {alpha}")
        return alpha

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, weights: dict[str, float]) -> dict[str, float]:

        """Validate non-empty weights and enforce non-negative long-only weights."""
        if not weights:
            raise ValueError("Portfolio weights dictionary cannot be empty.")
        total = sum(weights.values())
        if abs(total) < 1e-6:
            raise ValueError("Sum of weights cannot be zero.")
        for ticker, w in weights.items():
            if w < -1e-6:
                raise ValueError(f"Negative weight {w} for ticker '{ticker}' is not permitted.")
        return weights


class PortfolioVaRResponse(BaseModel):
    """Response model returning portfolio VaR, ES, and regulatory backtest diagnostics."""

    model_config = ConfigDict(extra="forbid")

    date: str = Field(description="Forecast effective date (YYYY-MM-DD).")
    alpha: float = Field(description="Target risk quantile.")
    var_return_pct: float = Field(
        description="Value-at-Risk expressed as a return percentage (negative value)."
    )
    var_inr: float = Field(
        description="Value-at-Risk in Indian Rupees (positive loss amount)."
    )
    expected_shortfall_pct: float = Field(
        description="Approximated Expected Shortfall as a return percentage (negative value)."
    )
    expected_shortfall_inr: float = Field(
        description="Approximated Expected Shortfall in Indian Rupees (positive loss amount)."
    )
    empirical_coverage_250d: float = Field(
        description="Observed empirical coverage over the most recent 250 trading days."
    )
    kupiec_p_value: float = Field(
        description="P-value of Kupiec Proportion of Failures test over 250 days."
    )
    christoffersen_ind_p_value: float = Field(
        description="P-value of Christoffersen Markov Independence test over 250 days."
    )
    basel_traffic_light: Literal["GREEN", "YELLOW", "RED"] = Field(
        description="Basel Committee Traffic Light status classification."
    )
    cache_hit: bool = Field(description="Whether response was served from cache.")
    latency_ms: float = Field(description="Execution latency in milliseconds.")


class HealthResponse(BaseModel):
    """Service health status and data freshness report."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="ok")
    redis_connected: bool = Field(description="Whether Redis connection is active.")
    cache_backend: Literal["redis", "in_memory_lru"] = Field(
        description="Active caching mechanism."
    )
    latest_market_date: str | None = Field(
        default=None,
        description="Latest date present in cached returns dataset.",
    )
    available_tickers: list[str] = Field(default_factory=list)


class RegimeStressBenchmarkResponse(BaseModel):
    """Precomputed regime stress test benchmark results."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    total_records: int
    data: list[dict[str, Any]]
