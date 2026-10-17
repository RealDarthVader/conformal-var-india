"""Portfolio Risk Engine with Conformal VaR, Covariance VaR, and Conformal Expected Shortfall.

Implements:
1. Portfolio return series aggregation from log-returns:
   R_{p,t} = ln(sum_{i=1}^k w_i * exp(r_{i,t}))
2. 1-day-ahead Portfolio VaR using Adaptive Conformal Inference (ACI).
3. Parametric Covariance-Matrix VaR benchmark:
   sigma_p,t = sqrt(w^T * Sigma_t * w), VaR_t(alpha) = Phi^{-1}(alpha) * sigma_p,t
4. Approximated Conformal Expected Shortfall (ES_alpha):
   Approximated by integrating ACI-calibrated VaR across a 10-point tail quantile grid:
   tau in {alpha/10, 2*alpha/10, ..., alpha}:
   ES_t(alpha) approx (1/10) * sum_{j=1}^{10} VaR_t^{ACI}(j * alpha / 10)
5. Strict sub-additivity & diversification metrics:
   Diversification Benefit = sum(w_i * VaR_{i,t}) - VaR_{p,t} >= 0
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from conformal_var.conformal.calibrator import AdaptiveConformalVaR
from conformal_var.models.baselines import HistoricalSimulationVaR

logger = logging.getLogger("portfolio_engine")


@dataclass(frozen=True)
class PortfolioRiskForecast:
    """Out-of-sample portfolio risk forecast for a specific trading day."""

    date: pd.Timestamp | str
    alpha: float
    portfolio_var_aci: float
    portfolio_es_aci: float
    parametric_cov_var: float
    individual_vars_weighted_sum: float
    diversification_benefit: float
    portfolio_return: float | None = None
    violation: bool | None = None


class PortfolioRiskEngine:
    """Uncertainty-calibrated portfolio risk engine for Indian equities."""

    def __init__(
        self,
        weights: Mapping[str, float],
        allow_short: bool = False,
        rolling_cov_window: int = 252,
        aci_window: int = 252,
        aci_gamma: float = 0.01,
        min_history: int = 252,
    ) -> None:
        """Initialize engine with ticker weights and rolling risk configuration.

        Parameters
        ----------
        weights : Mapping[str, float]
            Dictionary of {ticker: weight}.
        allow_short : bool
            If False, enforces w_i >= 0.
        rolling_cov_window : int
            Rolling window for sample covariance matrix estimation.
        aci_window : int
            Window for conformal calibration buffer.
        aci_gamma : float
            ACI step adaptation rate.
        min_history : int
            Minimum historical days needed before first out-of-sample prediction.
        """
        self.weights = self._validate_and_normalize_weights(weights, allow_short=allow_short)
        self.tickers = list(self.weights.keys())
        self.w_vec = np.array([self.weights[t] for t in self.tickers], dtype=np.float64)
        self.allow_short = allow_short
        self.rolling_cov_window = rolling_cov_window
        self.aci_window = aci_window
        self.aci_gamma = aci_gamma
        self.min_history = min_history

    @staticmethod
    def _validate_and_normalize_weights(
        weights: Mapping[str, float],
        allow_short: bool = False,
    ) -> dict[str, float]:
        """Validate ticker weights and normalize to sum to 1.0."""
        if not weights:
            raise ValueError("Portfolio weights dictionary cannot be empty.")

        clean_weights: dict[str, float] = {}
        for ticker, w in weights.items():
            val = float(w)
            if not allow_short and val < -1e-8:
                raise ValueError(
                    f"Long-only portfolio does not permit negative weight {val} for {ticker}."
                )
            clean_weights[ticker] = val

        total_weight = sum(clean_weights.values())
        if abs(total_weight) < 1e-8:
            raise ValueError(f"Sum of portfolio weights is too close to zero ({total_weight}).")

        # Normalize weights so sum w_i = 1.0
        normalized = {t: w / total_weight for t, w in clean_weights.items()}
        return normalized

    def aggregate_portfolio_returns(self, asset_returns: pd.DataFrame) -> pd.Series:
        """Construct historical portfolio log returns.

        R_{p,t} = ln(sum_{i=1}^k w_i * exp(r_{i,t}))
        """
        missing = [t for t in self.tickers if t not in asset_returns.columns]
        if missing:
            raise KeyError(f"Asset returns missing tickers: {missing}")

        sub_df = asset_returns[self.tickers].dropna()
        if len(sub_df) == 0:
            raise ValueError("No common overlapping dates among portfolio assets.")

        # Compute simple returns R_i = exp(r_i) - 1, portfolio simple return R_p = sum(w_i * R_i)
        # Portfolio log return r_p = ln(1 + R_p) = ln(sum(w_i * exp(r_i)))
        exp_r = np.exp(sub_df[self.tickers].to_numpy(dtype=np.float64))
        weighted_exp = np.dot(exp_r, self.w_vec)

        # Check positive argument
        if np.any(weighted_exp <= 0):
            raise ValueError("Portfolio total value factor <= 0 encountered in return aggregation.")

        port_log_ret = np.log(weighted_exp)
        return pd.Series(port_log_ret, index=sub_df.index, name="Portfolio_Return")

    def compute_parametric_covariance_var(
        self,
        asset_returns: pd.DataFrame,
        alpha: float = 0.05,
    ) -> pd.Series:
        """Compute rolling parametric covariance-matrix Value-at-Risk.

        sigma_{p,t}^2 = w^T * Sigma_{t-1} * w
        VaR_{p,t}(alpha) = Phi^{-1}(alpha) * sigma_{p,t}
        """
        sub_df = asset_returns[self.tickers].dropna()
        n_obs = len(sub_df)
        if n_obs <= self.rolling_cov_window:
            raise ValueError(
                f"Need > {self.rolling_cov_window} observations for covariance matrix, got {n_obs}"
            )

        z_alpha = norm.ppf(alpha)  # Negative value for alpha < 0.5
        var_preds = np.full(n_obs, np.nan, dtype=np.float64)

        # Causal rolling covariance: cov matrix up to t-1 used to forecast day t
        returns_mat = sub_df[self.tickers].to_numpy(dtype=np.float64)
        for t in range(self.rolling_cov_window, n_obs):
            past_window = returns_mat[t - self.rolling_cov_window : t]
            cov_mat = np.cov(past_window, rowvar=False)
            port_var = float(np.dot(self.w_vec, np.dot(cov_mat, self.w_vec)))
            port_vol = np.sqrt(max(0.0, port_var))
            var_preds[t] = z_alpha * port_vol

        return pd.Series(var_preds, index=sub_df.index, name=f"Parametric_VaR_{alpha}")

    def compute_conformal_expected_shortfall(
        self,
        quantile_forecasters: Sequence[AdaptiveConformalVaR],
        tail_grid: Sequence[float],
        base_quantiles: Sequence[float],
    ) -> float:
        """Compute Approximated Conformal Expected Shortfall across a 10-point grid.

        ES_t(alpha) approx (1/M) * sum_{j=1}^M VaR_t^{ACI}(tau_j)
        where tau_j in {alpha/10, ..., alpha}.
        """
        if len(quantile_forecasters) != len(tail_grid) or len(base_quantiles) != len(tail_grid):
            raise ValueError("Mismatched forecasters, grid points, or base quantiles length.")

        calibrated_vars: list[float] = []
        for calibrator, base_q in zip(quantile_forecasters, base_quantiles, strict=True):
            calibrated_vars.append(calibrator.predict_var(base_q))

        # Expected shortfall is the average of tail quantiles
        es_val = float(np.mean(calibrated_vars))
        return es_val

    def run_portfolio_risk_simulation(
        self,
        asset_returns: pd.DataFrame,
        alpha: float = 0.05,
        grid_points: int = 10,
    ) -> pd.DataFrame:
        """Run full out-of-sample portfolio risk walk-forward evaluation.

        Computes:
        - Portfolio log return R_{p,t}
        - Parametric Covariance VaR
        - ACI Calibrated Portfolio VaR(alpha)
        - 10-point grid Approximated Conformal Expected Shortfall ES(alpha)
        - Individual asset VaR weighted sum sum(w_i * VaR_i)
        - Diversification benefit sum(w_i * VaR_i) - VaR_{p,t} >= 0
        """
        # 1. Aggregate portfolio log-returns
        port_returns = self.aggregate_portfolio_returns(asset_returns)
        sub_assets = asset_returns[self.tickers].loc[port_returns.index]
        t_total = len(port_returns)

        start_idx = max(self.min_history, self.rolling_cov_window, self.aci_window)
        if t_total <= start_idx:
            raise ValueError(f"Need > {start_idx} days of history, got {t_total}")

        # 2. Parametric VaR
        param_var = self.compute_parametric_covariance_var(sub_assets, alpha=alpha)

        # 3. Setup Tail Quantile Grid for Approximated ES
        # tau in {alpha/10, 2*alpha/10, ..., alpha}
        tail_grid = [round((j * alpha) / grid_points, 6) for j in range(1, grid_points + 1)]

        # Forecasters for portfolio: HS baseline as base quantile + ACI calibrator
        base_forecasters = [HistoricalSimulationVaR(window=self.min_history) for _ in tail_grid]
        aci_calibrators = [
            AdaptiveConformalVaR(alpha=tau, gamma=self.aci_gamma, window=self.aci_window)
            for tau in tail_grid
        ]

        # Forecasters for individual assets at target alpha
        asset_forecasters = {
            t: HistoricalSimulationVaR(window=self.min_history) for t in self.tickers
        }
        asset_calibrators = {
            t: AdaptiveConformalVaR(alpha=alpha, gamma=self.aci_gamma, window=self.aci_window)
            for t in self.tickers
        }

        # Initialize calibration buffers using initial pre-sample [0 : start_idx]
        init_port_rets = port_returns.iloc[:start_idx]
        pre_start = max(10, self.min_history // 2)
        # Dynamic window slice min(t_idx, self.min_history)
        for tau, aci_c in zip(tail_grid, aci_calibrators, strict=True):
            # Pre-sample predictions using historical quantiles
            preds_init: list[float] = []
            for t_idx in range(pre_start, start_idx):
                h = init_port_rets.iloc[:t_idx].to_numpy()
                preds_init.append(float(np.quantile(h, tau)))
            aci_c.calibrate(
                np.array(preds_init),
                init_port_rets.iloc[pre_start:start_idx].to_numpy(),
            )

        for ticker in self.tickers:
            init_asset_rets = sub_assets[ticker].iloc[:start_idx]
            preds_asset_init: list[float] = []
            for t_idx in range(pre_start, start_idx):
                h = init_asset_rets.iloc[:t_idx].to_numpy()
                preds_asset_init.append(float(np.quantile(h, alpha)))
            asset_calibrators[ticker].calibrate(
                np.array(preds_asset_init),
                init_asset_rets.iloc[pre_start:start_idx].to_numpy(),
            )

        # 4. Out-of-sample walk-forward loop
        records: list[dict[str, Any]] = []

        test_indices = range(start_idx, t_total)
        for t in test_indices:
            dt = port_returns.index[t]
            r_p_realized = float(port_returns.iloc[t])

            # Causal history strictly up to t-1
            past_port = port_returns.iloc[:t]

            # A. Base quantiles for grid points
            grid_base_preds: list[float] = []
            for base_f, tau in zip(base_forecasters, tail_grid, strict=True):
                base_f.fit(past_port)
                grid_base_preds.append(base_f.predict_var(tau))

            # Portfolio ACI VaR(alpha) is the last grid point (tau = alpha)
            port_var_aci = aci_calibrators[-1].predict_var(grid_base_preds[-1])

            # Approximated Conformal Expected Shortfall across 10 points
            port_es_aci = self.compute_conformal_expected_shortfall(
                aci_calibrators,
                tail_grid,
                grid_base_preds,
            )

            # Enforce statistical identity ES_t(alpha) <= VaR_t(alpha)
            # Since ES integrates deeper into the left tail, its negative value is <= VaR
            port_es_aci = min(port_es_aci, port_var_aci)

            # B. Individual asset VaRs at target alpha
            indiv_vars: list[float] = []
            for ticker in self.tickers:
                past_asset = sub_assets[ticker].iloc[:t]
                a_f = asset_forecasters[ticker]
                a_f.fit(past_asset)
                raw_indiv_q = a_f.predict_var(alpha)
                calib_indiv_v = asset_calibrators[ticker].predict_var(raw_indiv_q)
                indiv_vars.append(calib_indiv_v)

            weighted_sum_indiv = float(np.dot(self.w_vec, np.array(indiv_vars)))

            # Diversification benefit: sum(w_i * VaR_i) - VaR_{p}
            # Note: VaRs are negative returns.
            # Diversification makes portfolio risk less negative (closer to zero):
            # VaR_p >= sum(w_i * VaR_i)  ==>  |VaR_p| <= sum(w_i * |VaR_i|)
            # In absolute terms: Diversification Benefit = sum(w_i * |VaR_i|) - |VaR_p|
            div_benefit_abs = float(np.dot(self.w_vec, np.abs(np.array(indiv_vars)))) - abs(
                port_var_aci
            )

            p_cov_var = float(param_var.iloc[t])

            violation = int(r_p_realized < port_var_aci)

            records.append(
                {
                    "date": dt,
                    "portfolio_return": r_p_realized,
                    "portfolio_var_aci": port_var_aci,
                    "portfolio_es_aci": port_es_aci,
                    "parametric_cov_var": p_cov_var,
                    "weighted_indiv_var": weighted_sum_indiv,
                    "diversification_benefit": div_benefit_abs,
                    "violation": violation,
                }
            )

            # Post-close online ACI updates with realized return
            for aci_c, base_q in zip(aci_calibrators, grid_base_preds, strict=True):
                aci_c.update(base_q, r_p_realized)

            for ticker in self.tickers:
                r_asset_realized = float(sub_assets[ticker].iloc[t])
                a_f = asset_forecasters[ticker]
                raw_q = a_f.predict_var(alpha)
                asset_calibrators[ticker].update(raw_q, r_asset_realized)

        results_df = pd.DataFrame(records).set_index("date")
        return results_df
