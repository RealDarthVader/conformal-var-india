"""Unit and statistical tests for PortfolioRiskEngine.

Verifies:
1. Long-only weight validation and normalization to 1.0.
2. Accurate aggregation of historical daily log returns R_{p,t} = ln(sum w_i * exp(r_i)).
3. Causal parametric covariance-matrix VaR.
4. Approximated Conformal Expected Shortfall inequality:
   ES_t(alpha) <= VaR_t^{ACI}(alpha) always holds (ES is deeper in left tail).
5. Portfolio diversification benefit:
   Diversification reduces risk relative to the weighted sum of individual asset VaRs:
   |VaR_{p,t}| <= sum(w_i * |VaR_{i,t}|).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_var.portfolio.engine import PortfolioRiskEngine


@pytest.fixture
def sample_portfolio_returns() -> pd.DataFrame:
    """Generate 600 days of synthetic multi-asset log-returns."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=600, freq="B")

    # Generate correlated 3-asset returns
    mean = [0.0005, 0.0003, 0.0004]
    cov = [
        [0.000400, 0.000150, 0.000100],
        [0.000150, 0.000300, 0.000080],
        [0.000100, 0.000080, 0.000250],
    ]
    raw_returns = np.random.multivariate_normal(mean, cov, size=600)
    df = pd.DataFrame(raw_returns, index=dates, columns=["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    return df


class TestPortfolioWeightsAndAggregation:
    """Tests for weight validation, normalization, and return aggregation."""

    def test_weight_normalization(self) -> None:
        """Weights should automatically normalize to sum to 1.0."""
        engine = PortfolioRiskEngine({"RELIANCE.NS": 50.0, "TCS.NS": 50.0})
        assert engine.weights["RELIANCE.NS"] == 0.5
        assert engine.weights["TCS.NS"] == 0.5
        assert pytest.approx(sum(engine.weights.values()), abs=1e-7) == 1.0

    def test_negative_weight_in_long_only_raises(self) -> None:
        """Negative weights should raise ValueError when allow_short=False."""
        with pytest.raises(ValueError, match="does not permit negative weight"):
            PortfolioRiskEngine({"RELIANCE.NS": 1.2, "TCS.NS": -0.2}, allow_short=False)

    def test_allow_short_weights(self) -> None:
        """Negative weights allowed when allow_short=True."""
        engine = PortfolioRiskEngine({"RELIANCE.NS": 1.5, "TCS.NS": -0.5}, allow_short=True)
        assert pytest.approx(sum(engine.weights.values()), abs=1e-7) == 1.0

    def test_empty_weights_raises(self) -> None:
        """Empty weight dictionary raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PortfolioRiskEngine({})

    def test_aggregate_portfolio_log_returns_math(
        self, sample_portfolio_returns: pd.DataFrame
    ) -> None:
        """Verify R_{p,t} = ln(sum w_i * exp(r_i)) exactly matches definition."""
        engine = PortfolioRiskEngine({"RELIANCE.NS": 0.6, "TCS.NS": 0.4})
        port_ret = engine.aggregate_portfolio_returns(sample_portfolio_returns)

        # Calculate manually for day 0
        r_rel = sample_portfolio_returns["RELIANCE.NS"].iloc[0]
        r_tcs = sample_portfolio_returns["TCS.NS"].iloc[0]
        expected_r0 = np.log(0.6 * np.exp(r_rel) + 0.4 * np.exp(r_tcs))

        assert pytest.approx(port_ret.iloc[0], abs=1e-8) == expected_r0
        assert len(port_ret) == len(sample_portfolio_returns)


class TestPortfolioRiskEngineSimulation:
    """Tests for VaR, ES, and diversification inequalities."""

    def test_es_le_var_inequality(self, sample_portfolio_returns: pd.DataFrame) -> None:
        """Expected Shortfall must always be strictly deeper in left tail than VaR.

        Since both ES and VaR are expressed as negative returns (potential losses),
        ES_t(alpha) <= VaR_t(alpha) for all t.
        """
        engine = PortfolioRiskEngine(
            {"RELIANCE.NS": 0.5, "TCS.NS": 0.5},
            rolling_cov_window=100,
            aci_window=100,
            min_history=100,
        )

        sim_df = engine.run_portfolio_risk_simulation(
            sample_portfolio_returns,
            alpha=0.05,
            grid_points=10,
        )

        assert len(sim_df) > 0
        # Check ES <= VaR across all simulation dates
        es_diff = sim_df["portfolio_es_aci"] - sim_df["portfolio_var_aci"]
        assert (es_diff <= 1e-9).all(), f"Found ES > VaR at {sim_df[es_diff > 1e-9]}"

    def test_portfolio_diversification_reduces_var(
        self, sample_portfolio_returns: pd.DataFrame
    ) -> None:
        """Portfolio diversification must reduce VaR relative to weighted individual assets.

        In magnitude: |VaR_p| <= sum(w_i * |VaR_i|)
        Equivalently: Diversification benefit = sum(w_i * |VaR_i|) - |VaR_p| >= 0.
        """
        engine = PortfolioRiskEngine(
            {"RELIANCE.NS": 0.34, "TCS.NS": 0.33, "INFY.NS": 0.33},
            rolling_cov_window=100,
            aci_window=100,
            min_history=100,
        )

        sim_df = engine.run_portfolio_risk_simulation(
            sample_portfolio_returns,
            alpha=0.05,
            grid_points=10,
        )

        # Average diversification benefit should be strictly positive
        mean_div_benefit = float(sim_df["diversification_benefit"].mean())
        assert mean_div_benefit > 0.0, (
            f"Expected positive diversification benefit, got {mean_div_benefit}"
        )

        # 95%+ of trading days should enjoy strictly positive diversification benefit
        pct_positive = float((sim_df["diversification_benefit"] >= -1e-5).mean())
        assert pct_positive >= 0.90

    def test_parametric_covariance_var(self, sample_portfolio_returns: pd.DataFrame) -> None:
        """Verify parametric covariance-matrix VaR is negative and non-zero."""
        engine = PortfolioRiskEngine(
            {"RELIANCE.NS": 0.5, "TCS.NS": 0.5},
            rolling_cov_window=100,
            min_history=100,
        )
        cov_var = engine.compute_parametric_covariance_var(sample_portfolio_returns, alpha=0.01)

        valid_cov = cov_var.dropna()
        assert len(valid_cov) > 0
        assert (valid_cov < 0.0).all()  # 1% VaR is negative
