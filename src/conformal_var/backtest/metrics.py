"""Statistical hypothesis tests and regulatory backtest metrics for Value-at-Risk.

Implements bank market-risk desk standards:
1. Kupiec Proportion of Failures (POF) Test (Unconditional Coverage LR_uc ~ chi2_1).
2. Christoffersen Markov Independence Test (Violation Clustering LR_ind ~ chi2_1).
3. Christoffersen Conditional Coverage Joint Test (LR_cc = LR_uc + LR_ind ~ chi2_2).
4. Basel Committee Traffic-Light Classification (Green, Yellow, Red zones with multiplier k).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import chi2


def _to_binary_array(
    violations: Sequence[int | float | bool] | npt.NDArray[Any] | pd.Series,
) -> npt.NDArray[np.int64]:
    """Convert input sequence into a 1D binary numpy array of 0s and 1s."""
    if isinstance(violations, pd.Series):
        arr = violations.dropna().to_numpy()
    else:
        arr = np.asarray(violations)
    out: npt.NDArray[np.int64] = np.asarray(arr > 0, dtype=np.int64)
    return out


@dataclass(frozen=True)
class KupiecResult:
    """Result of Kupiec Proportion of Failures (POF) test."""

    statistic: float
    p_value: float
    passed: bool
    violations: int
    total_days: int
    empirical_rate: float
    alpha: float


@dataclass(frozen=True)
class ChristoffersenIndependenceResult:
    """Result of Christoffersen Markov first-order independence test."""

    statistic: float
    p_value: float
    passed: bool
    n00: int
    n01: int
    n10: int
    n11: int
    pi01: float
    pi11: float
    pi: float


@dataclass(frozen=True)
class ChristoffersenConditionalCoverageResult:
    """Result of Christoffersen Joint Conditional Coverage test."""

    statistic: float
    p_value: float
    passed: bool
    lr_uc: float
    lr_ind: float


@dataclass(frozen=True)
class TrafficLightResult:
    """Result of Basel Committee on Banking Supervision Traffic Light test."""

    zone: str
    multiplier: float
    violations: int
    scaled_violations: int
    status: str


@dataclass(frozen=True)
class BacktestReport:
    """Comprehensive backtest report aggregating all desk tests."""

    alpha: float
    total_days: int
    violations: int
    kupiec: KupiecResult
    independence: ChristoffersenIndependenceResult
    conditional_coverage: ChristoffersenConditionalCoverageResult
    traffic_light: TrafficLightResult

    def to_dict(self) -> dict[str, Any]:
        """Convert report metrics into a dictionary."""
        return {
            "alpha": self.alpha,
            "total_days": self.total_days,
            "violations": self.violations,
            "empirical_rate": f"{self.kupiec.empirical_rate:.4f}",
            "kupiec_lr": round(self.kupiec.statistic, 4),
            "kupiec_p_value": round(self.kupiec.p_value, 4),
            "kupiec_passed": self.kupiec.passed,
            "independence_lr": round(self.independence.statistic, 4),
            "independence_p_value": round(self.independence.p_value, 4),
            "independence_passed": self.independence.passed,
            "conditional_coverage_lr": round(self.conditional_coverage.statistic, 4),
            "conditional_coverage_p_value": round(self.conditional_coverage.p_value, 4),
            "conditional_coverage_passed": self.conditional_coverage.passed,
            "basel_zone": self.traffic_light.zone,
            "basel_multiplier": self.traffic_light.multiplier,
            "basel_status": self.traffic_light.status,
        }


def kupiec_pof_test(
    violations: Sequence[int | float | bool] | npt.NDArray[Any] | pd.Series,
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> KupiecResult:
    """Execute Kupiec Proportion of Failures (POF) test for unconditional coverage.

    LR_uc = -2 * ln[ ((1 - alpha)^(T - N) * alpha^N) / ((1 - p_hat)^(T - N) * p_hat^N) ] ~ chi2(1)
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    v_arr = _to_binary_array(violations)
    t_days = len(v_arr)
    if t_days == 0:
        raise ValueError("Violations array cannot be empty.")

    n_viol = int(np.sum(v_arr))
    p_hat = n_viol / t_days

    # Calculate log-likelihoods handling edge cases N=0 and N=T
    if n_viol == 0:
        ln_l1 = 0.0
        ln_l0 = float(t_days * np.log(1.0 - alpha))
    elif n_viol == t_days:
        ln_l1 = 0.0
        ln_l0 = float(t_days * np.log(alpha))
    else:
        ln_l1 = float((t_days - n_viol) * np.log(1.0 - p_hat) + n_viol * np.log(p_hat))
        ln_l0 = float((t_days - n_viol) * np.log(1.0 - alpha) + n_viol * np.log(alpha))

    lr_uc = max(0.0, 2.0 * (ln_l1 - ln_l0))
    p_value = float(chi2.sf(lr_uc, df=1))
    crit_alpha = 1.0 - confidence
    passed = p_value >= crit_alpha

    return KupiecResult(
        statistic=lr_uc,
        p_value=p_value,
        passed=passed,
        violations=n_viol,
        total_days=t_days,
        empirical_rate=p_hat,
        alpha=alpha,
    )


def christoffersen_independence_test(
    violations: Sequence[int | float | bool] | npt.NDArray[Any] | pd.Series,
    confidence: float = 0.95,
) -> ChristoffersenIndependenceResult:
    """Execute Christoffersen Markov test for violation independence / clustering.

    LR_ind = -2 * ln[ L(pi) / L(pi_01, pi_11) ] ~ chi2(1)
    """
    v_arr = _to_binary_array(violations)
    t_days = len(v_arr)
    if t_days < 2:
        raise ValueError(f"Need at least 2 observations for independence test, got {t_days}")

    # First-order transitions
    i_prev = v_arr[:-1]
    i_curr = v_arr[1:]

    n00 = int(np.sum((i_prev == 0) & (i_curr == 0)))
    n01 = int(np.sum((i_prev == 0) & (i_curr == 1)))
    n10 = int(np.sum((i_prev == 1) & (i_curr == 0)))
    n11 = int(np.sum((i_prev == 1) & (i_curr == 1)))

    denom0 = n00 + n01
    denom1 = n10 + n11
    pi01 = float(n01 / denom0) if denom0 > 0 else 0.0
    pi11 = float(n11 / denom1) if denom1 > 0 else 0.0
    total_trans = t_days - 1
    pi = float((n01 + n11) / total_trans) if total_trans > 0 else 0.0

    # Log-likelihood under alternative (Markov dependence)
    ln_l1_parts: list[float] = []
    if n00 > 0 and (1.0 - pi01) > 0:
        ln_l1_parts.append(n00 * np.log(1.0 - pi01))
    if n01 > 0 and pi01 > 0:
        ln_l1_parts.append(n01 * np.log(pi01))
    if n10 > 0 and (1.0 - pi11) > 0:
        ln_l1_parts.append(n10 * np.log(1.0 - pi11))
    if n11 > 0 and pi11 > 0:
        ln_l1_parts.append(n11 * np.log(pi11))
    ln_l1 = float(np.sum(ln_l1_parts)) if ln_l1_parts else 0.0

    # Log-likelihood under null (independent Bernoulli trials)
    ln_l0_parts: list[float] = []
    if (n00 + n10) > 0 and (1.0 - pi) > 0:
        ln_l0_parts.append((n00 + n10) * np.log(1.0 - pi))
    if (n01 + n11) > 0 and pi > 0:
        ln_l0_parts.append((n01 + n11) * np.log(pi))
    ln_l0 = float(np.sum(ln_l0_parts)) if ln_l0_parts else 0.0

    lr_ind = max(0.0, 2.0 * (ln_l1 - ln_l0))
    p_value = float(chi2.sf(lr_ind, df=1))
    crit_alpha = 1.0 - confidence
    passed = p_value >= crit_alpha

    return ChristoffersenIndependenceResult(
        statistic=lr_ind,
        p_value=p_value,
        passed=passed,
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
        pi01=pi01,
        pi11=pi11,
        pi=pi,
    )


def christoffersen_conditional_coverage_test(
    violations: Sequence[int | float | bool] | npt.NDArray[Any] | pd.Series,
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> ChristoffersenConditionalCoverageResult:
    """Execute Christoffersen Joint Conditional Coverage test.

    LR_cc = LR_uc + LR_ind ~ chi2(2)
    """
    kupiec = kupiec_pof_test(violations, alpha=alpha, confidence=confidence)
    ind = christoffersen_independence_test(violations, confidence=confidence)

    lr_cc = kupiec.statistic + ind.statistic
    p_value = float(chi2.sf(lr_cc, df=2))
    crit_alpha = 1.0 - confidence
    passed = p_value >= crit_alpha

    return ChristoffersenConditionalCoverageResult(
        statistic=lr_cc,
        p_value=p_value,
        passed=passed,
        lr_uc=kupiec.statistic,
        lr_ind=ind.statistic,
    )


def basel_traffic_light(
    violations: Sequence[int | float | bool] | npt.NDArray[Any] | pd.Series,
    window_size: int = 250,
) -> TrafficLightResult:
    """Basel Committee on Banking Supervision (BCBS) Traffic Light Classification.

    For 99% VaR (alpha=0.01) evaluated over a standard 250-trading-day window:
    - Green Zone (0 to 4 violations): k = 3.00 (Model Accepted)
    - Yellow Zone (5 to 9 violations): k in [3.40, 3.85] (Capital Surcharge)
    - Red Zone (>= 10 violations): k = 4.00 (Model Rejected)
    """
    v_arr = _to_binary_array(violations)
    n_raw = int(np.sum(v_arr))
    t_days = len(v_arr)

    if t_days == 0:
        raise ValueError("Violations array cannot be empty.")

    # Scale violations to standard 250-day window
    scaled_n = int(round(n_raw * (window_size / t_days))) if t_days != window_size else n_raw

    # Assign Basel traffic light zone and supervisory capital multiplier
    if scaled_n <= 4:
        zone = "Green"
        multiplier = 3.00
        status = "Model Accepted"
    elif scaled_n == 5:
        zone = "Yellow"
        multiplier = 3.40
        status = "Warning: Capital Surcharge Added"
    elif scaled_n == 6:
        zone = "Yellow"
        multiplier = 3.50
        status = "Warning: Capital Surcharge Added"
    elif scaled_n == 7:
        zone = "Yellow"
        multiplier = 3.65
        status = "Warning: Capital Surcharge Added"
    elif scaled_n == 8:
        zone = "Yellow"
        multiplier = 3.75
        status = "Warning: Capital Surcharge Added"
    elif scaled_n == 9:
        zone = "Yellow"
        multiplier = 3.85
        status = "Warning: Capital Surcharge Added"
    else:
        zone = "Red"
        multiplier = 4.00
        status = "Model Rejected (Regulatory Invalidation)"

    return TrafficLightResult(
        zone=zone,
        multiplier=multiplier,
        violations=n_raw,
        scaled_violations=scaled_n,
        status=status,
    )


def run_full_backtest(
    violations: Sequence[int | float | bool] | npt.NDArray[Any] | pd.Series,
    alpha: float = 0.05,
    confidence: float = 0.95,
    window_size: int = 250,
) -> BacktestReport:
    """Execute complete risk-desk backtest battery."""
    v_arr = _to_binary_array(violations)
    kupiec = kupiec_pof_test(v_arr, alpha=alpha, confidence=confidence)
    ind = christoffersen_independence_test(v_arr, confidence=confidence)
    cc = christoffersen_conditional_coverage_test(v_arr, alpha=alpha, confidence=confidence)
    tl = basel_traffic_light(v_arr, window_size=window_size)

    return BacktestReport(
        alpha=alpha,
        total_days=len(v_arr),
        violations=int(np.sum(v_arr)),
        kupiec=kupiec,
        independence=ind,
        conditional_coverage=cc,
        traffic_light=tl,
    )
