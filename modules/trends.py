"""
Trend-line helpers shared by the time-series and phenology plots in tab 3:
a plain linear regression (with a slope you can print on the plot) and a
smoothed GAM curve (a single spline term via pygam, the closest Python
equivalent of the R app's original `geom_smooth(method="gam", formula = y
~ s(x, k=4))`).

Both take a numeric x (not a date) and a y array, so callers are
responsible for converting dates to e.g. days-since-first-scene first -
`days_since` below does that conversion consistently for both trend types
and for the slope's units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


def days_since(datetimes: pd.Series) -> np.ndarray:
    """Converts a datetime Series to float days since its own minimum -
    the x-axis used internally by both trend fits (dates themselves aren't
    numeric)."""
    dt = pd.to_datetime(datetimes)
    return (dt - dt.min()).dt.total_seconds().values / 86400.0


@dataclass
class LinearTrendResult:
    x: np.ndarray
    y: np.ndarray
    slope_per_day: float
    intercept: float
    slope_per_year: float
    r2: float


def linear_trend(x_days: np.ndarray, y: np.ndarray) -> Optional[LinearTrendResult]:
    """Ordinary least-squares fit of y ~ x_days. Returns None if there
    aren't at least 2 finite points (can't fit a line through fewer)."""
    mask = np.isfinite(x_days) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    x, y = x_days[mask], y[mask]
    if np.ptp(x) == 0:
        return None  # all points on the same day - slope undefined
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    order = np.argsort(x)
    return LinearTrendResult(
        x=x[order], y=fitted[order],
        slope_per_day=float(slope), intercept=float(intercept),
        slope_per_year=float(slope * DAYS_PER_YEAR), r2=float(r2),
    )


def gam_trend(x_days: np.ndarray, y: np.ndarray, n_splines: Optional[int] = None, n_points: int = 150):
    """Fits a single-term GAM (pygam) of y ~ s(x_days). Returns
    (x_grid, y_pred, error) - x_grid/y_pred are ready to plot as a smooth
    curve and error is None on success; on failure x_grid/y_pred are None
    and error is a short string saying why. `n_splines` is auto-scaled to
    sample size if not given, capped to keep the curve from chasing
    individual points on small samples.

    The "why" matters here: pygam's own `.fit()` can raise on genuinely
    degenerate inputs, but it can also raise for reasons that have nothing
    to do with "not enough data" - e.g. a pygam/scipy version mismatch in
    whatever environment this is running in. Earlier versions of this
    function caught *any* exception and reported it identically to "fewer
    than 5 points", which is actively misleading when there are hundreds of
    points and the real problem is something else - see this function's
    call sites in app.py, which now show `error` directly instead of a
    generic message whenever it's set."""
    mask = np.isfinite(x_days) & np.isfinite(y)
    if mask.sum() < 5:
        return None, None, "fewer than 5 usable (finite) points"
    x = np.asarray(x_days[mask], dtype="float64")
    y = np.asarray(y[mask], dtype="float64")
    if np.ptp(x) == 0:
        return None, None, "all usable points fall on the same day"
    if n_splines is None:
        n_splines = int(np.clip(mask.sum() // 4, 4, 12))
    try:
        from pygam import LinearGAM, s

        gam = LinearGAM(s(0, n_splines=n_splines)).fit(x.reshape(-1, 1), y)
        x_grid = np.linspace(x.min(), x.max(), n_points)
        y_pred = gam.predict(x_grid.reshape(-1, 1))
        return x_grid, y_pred, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"
