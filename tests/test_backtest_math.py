"""Analytical and statistical verification tests for Value-at-Risk backtest metrics.

Tests exact mathematical properties of:
1. Kupiec Proportion of Failures (POF) LR_uc test.
2. Christoffersen Markov first-order independence LR_ind test.
3. Christoffersen Joint Conditional Coverage LR_cc test.
4. Basel Committee Traffic-Light classification boundaries & capital multipliers.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_var.backtest.metrics import (
    basel_traffic_light,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    kupiec_pof_test,
    run_full_backtest,
)


class TestKupiecPOFMath:
    """Analytical verification of Kupiec POF test."""

    def test_exact_theoretical_rate_matches_alpha(self) -> None:
        """When empirical failure rate matches target alpha exactly, LR_uc must be 0.0."""
        # T = 1000, alpha = 0.01, N = 10 -> p_hat = 0.01
        violations = np.zeros(1000, dtype=int)
        violations[np.linspace(10, 990, 10, dtype=int)] = 1

        result = kupiec_pof_test(violations, alpha=0.01, confidence=0.95)

        assert result.violations == 10
        assert result.total_days == 1000
        assert pytest.approx(result.empirical_rate, rel=1e-6) == 0.01
        assert pytest.approx(result.statistic, abs=1e-9) == 0.0
        assert pytest.approx(result.p_value, rel=1e-5) == 1.0
        assert result.passed is True

    def test_alpha_005_exact_rate(self) -> None:
        """When T=1000, alpha=0.05, N=50, LR_uc should be 0.0."""
        violations = np.zeros(1000, dtype=int)
        violations[np.linspace(0, 980, 50, dtype=int)] = 1

        result = kupiec_pof_test(violations, alpha=0.05, confidence=0.95)
        assert result.violations == 50
        assert pytest.approx(result.empirical_rate, rel=1e-6) == 0.05
        assert pytest.approx(result.statistic, abs=1e-9) == 0.0
        assert pytest.approx(result.p_value, rel=1e-5) == 1.0
        assert result.passed is True

    def test_kupiec_rejection_on_excessive_failures(self) -> None:
        """Severe underestimation of risk leads to immediate rejection (p < 0.001)."""
        # T = 1000, alpha = 0.01, N = 40 (4x higher than expected)
        violations = np.zeros(1000, dtype=int)
        violations[:40] = 1

        result = kupiec_pof_test(violations, alpha=0.01, confidence=0.95)
        assert result.violations == 40
        assert result.passed is False
        assert result.p_value < 1e-4

    def test_edge_case_zero_violations(self) -> None:
        """Edge case N=0 handled gracefully without divide-by-zero or NaN."""
        violations = np.zeros(500, dtype=int)
        result = kupiec_pof_test(violations, alpha=0.01, confidence=0.95)

        assert result.violations == 0
        assert result.empirical_rate == 0.0
        assert np.isfinite(result.statistic)
        assert np.isfinite(result.p_value)

    def test_edge_case_all_violations(self) -> None:
        """Edge case N=T handled gracefully."""
        violations = np.ones(100, dtype=int)
        result = kupiec_pof_test(violations, alpha=0.01, confidence=0.95)

        assert result.violations == 100
        assert result.empirical_rate == 1.0
        assert np.isfinite(result.statistic)
        assert result.passed is False

    def test_invalid_alpha_raises(self) -> None:
        """Alpha outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            kupiec_pof_test(np.zeros(10), alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            kupiec_pof_test(np.zeros(10), alpha=1.0)


class TestChristoffersenIndependenceMath:
    """Analytical verification of Christoffersen Markov test."""

    def test_spaced_out_violations_pass_independence(self) -> None:
        """10 violations spaced out evenly across 1000 days show no clustering."""
        violations = np.zeros(1000, dtype=int)
        # Evenly spaced out: indices 50, 150, 250, ..., 950
        indices = np.arange(50, 1000, 100)
        violations[indices] = 1

        result = christoffersen_independence_test(violations, confidence=0.95)

        # In spaced violations, n11 must be 0
        assert result.n11 == 0
        assert result.n01 == 10
        assert result.n10 == 10
        assert result.n00 == 1000 - 1 - 20
        assert result.passed is True
        assert result.p_value >= 0.05

    def test_clustered_violations_fail_independence(self) -> None:
        """10 consecutive violations clustered together must strongly reject independence."""
        violations = np.zeros(1000, dtype=int)
        # 10 consecutive violations in the middle of the sample
        violations[500:510] = 1

        result = christoffersen_independence_test(violations, confidence=0.95)

        # Transitions: 9 instances of 1->1, 1 instance of 1->0, 1 instance of 0->1
        assert result.n11 == 9
        assert result.n10 == 1
        assert result.n01 == 1
        assert result.n00 == 1000 - 1 - 11

        # Very high likelihood ratio and extremely low p-value (p < 0.001)
        assert result.statistic > 50.0
        assert result.p_value < 1e-10
        assert result.passed is False

    def test_insufficient_sample_raises(self) -> None:
        """Fewer than 2 observations raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 observations"):
            christoffersen_independence_test(np.array([1]))


class TestChristoffersenConditionalCoverageMath:
    """Analytical verification of Joint Conditional Coverage LR_cc = LR_uc + LR_ind."""

    def test_additive_chi2_statistic(self) -> None:
        """LR_cc must exactly equal LR_uc + LR_ind."""
        np.random.seed(42)
        # Random binary sequence
        violations = (np.random.rand(1000) < 0.03).astype(int)

        kupiec = kupiec_pof_test(violations, alpha=0.01)
        ind = christoffersen_independence_test(violations)
        cc = christoffersen_conditional_coverage_test(violations, alpha=0.01)

        assert pytest.approx(cc.statistic, abs=1e-8) == kupiec.statistic + ind.statistic
        assert pytest.approx(cc.lr_uc, abs=1e-8) == kupiec.statistic
        assert pytest.approx(cc.lr_ind, abs=1e-8) == ind.statistic

    def test_clustered_sequence_fails_conditional_coverage(self) -> None:
        """Sequence with correct count (N=10/1000) but consecutive clustering fails joint test."""
        violations = np.zeros(1000, dtype=int)
        violations[100:110] = 1

        kupiec = kupiec_pof_test(violations, alpha=0.01)
        cc = christoffersen_conditional_coverage_test(violations, alpha=0.01)

        # Kupiec passes because failure count matches expected 1% exactly
        assert pytest.approx(kupiec.statistic, abs=1e-9) == 0.0
        assert kupiec.passed is True

        # Joint conditional coverage fails because of severe clustering
        assert cc.passed is False
        assert cc.p_value < 1e-10


class TestBaselTrafficLight:
    """Analytical verification of Basel Committee traffic light boundaries and multipliers."""

    @pytest.mark.parametrize(
        ("n_violations", "expected_zone", "expected_mult"),
        [
            (0, "Green", 3.00),
            (1, "Green", 3.00),
            (2, "Green", 3.00),
            (3, "Green", 3.00),
            (4, "Green", 3.00),
            (5, "Yellow", 3.40),
            (6, "Yellow", 3.50),
            (7, "Yellow", 3.65),
            (8, "Yellow", 3.75),
            (9, "Yellow", 3.85),
            (10, "Red", 4.00),
            (15, "Red", 4.00),
            (25, "Red", 4.00),
        ],
    )
    def test_basel_boundaries_250_days(
        self,
        n_violations: int,
        expected_zone: str,
        expected_mult: float,
    ) -> None:
        """Verify exact boundary behavior on standard 250-day window."""
        violations = np.zeros(250, dtype=int)
        if n_violations > 0:
            violations[:n_violations] = 1

        res = basel_traffic_light(violations, window_size=250)
        assert res.zone == expected_zone
        assert pytest.approx(res.multiplier, abs=1e-4) == expected_mult
        assert res.scaled_violations == n_violations

    def test_basel_scaling_for_different_lengths(self) -> None:
        """Verify scaling when sample length differs from 250 days."""
        # 1000 days with 16 violations -> 16 * (250 / 1000) = 4 violations -> Green
        violations = np.zeros(1000, dtype=int)
        violations[:16] = 1

        res = basel_traffic_light(violations, window_size=250)
        assert res.violations == 16
        assert res.scaled_violations == 4
        assert res.zone == "Green"
        assert res.multiplier == 3.00

        # 1000 days with 40 violations -> 40 * (250 / 1000) = 10 violations -> Red
        violations_red = np.zeros(1000, dtype=int)
        violations_red[:40] = 1
        res_red = basel_traffic_light(violations_red, window_size=250)
        assert res_red.scaled_violations == 10
        assert res_red.zone == "Red"
        assert res_red.multiplier == 4.00


class TestFullBacktestReport:
    """Verify aggregated desk report generator."""

    def test_full_backtest_integration(self) -> None:
        """Ensure full backtest battery computes all metrics and serializes to dict."""
        violations = np.zeros(500, dtype=int)
        violations[::50] = 1  # 10 violations

        report = run_full_backtest(violations, alpha=0.05, confidence=0.95)
        d = report.to_dict()

        assert "kupiec_lr" in d
        assert "independence_lr" in d
        assert "conditional_coverage_lr" in d
        assert "basel_zone" in d
        assert "basel_multiplier" in d
        assert d["basel_zone"] in {"Green", "Yellow", "Red"}
