"""Conformal calibration engines for finite-sample coverage guarantees."""

from conformal_var.conformal.calibrator import (
    AdaptiveConformalVaR,
    SlidingWindowConformalVaR,
    SplitConformalVaR,
    compute_conformity_residual,
    finite_sample_conformal_quantile,
)
from conformal_var.conformal.vol_scaled import (
    VolScaledConformalVaR,
    compute_vol_scaled_residual,
)

__all__ = [
    "AdaptiveConformalVaR",
    "SlidingWindowConformalVaR",
    "SplitConformalVaR",
    "VolScaledConformalVaR",
    "compute_conformity_residual",
    "compute_vol_scaled_residual",
    "finite_sample_conformal_quantile",
]
