"""
Spectral index / Tasselled Cap calculations, computed from surface
reflectance (0-1 fraction) numpy arrays.

DEA ARD `nbart_*` bands are delivered as scaled integers (surface
reflectance x 10000); callers should divide by 10000 before passing bands
in here (`analysis.py` does this).

Tasselled Cap coefficients are the Crist (1985) reflectance-factor
coefficients for Landsat TM bands 1,2,3,4,5,7 (blue, green, red, nir,
swir1, swir2) - the same coefficients used for wetness/brightness in this
project's original R Shiny app. Greenness uses the same source's published
coefficients. These coefficients are applied as-is to Sentinel-2's
equivalent bands (nir_1 -> nir, swir_2 -> swir1, swir_3 -> swir2); this is
an approximation also used in various Sentinel-2 tasselled-cap studies (the
underlying spectral responses differ from TM, but a like-for-like
Landsat-vs-Sentinel time series is more useful for trend continuity than
using two different coefficient sets that aren't directly comparable).

NDVI = (nir - red) / (nir + red)

NDSI (Normalized Difference Snow Index) = (green - swir1) / (green + swir1)
is computed separately by `compute_ndsi()` below, not by `compute_indices()`
- unlike the other four metrics, it needs its own, broader per-pixel
validity mask (one that doesn't exclude fmask's snow class, since the whole
point of NDSI here is to reveal snow-covered scenes the vegetation-index
mask would otherwise hide), so it takes its own `bands` dict built from
that separate mask. See `analysis.py`'s NDSI_VALID_FMASK_CLASSES.
"""

from __future__ import annotations

import numpy as np

# Crist (1985), Remote Sensing of Environment 17:301-306.
# order: blue, green, red, nir, swir1, swir2
TC_BRIGHTNESS = np.array([0.2043, 0.4158, 0.5524, 0.5741, 0.3124, -0.2303])
TC_GREENNESS = np.array([-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446])
TC_WETNESS = np.array([0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109])

REQUIRED_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
NDSI_REQUIRED_BANDS = ["green", "swir1"]

METRICS = ["NDVI", "TCB", "TCG", "TCW", "NDSI"]

METRIC_LABELS = {
    "NDVI": "NDVI (vegetation index)",
    "TCB": "Tasselled Cap Brightness",
    "TCG": "Tasselled Cap Greenness",
    "TCW": "Tasselled Cap Wetness",
    "NDSI": "NDSI (snow index)",
}


def _tc(bands: dict, coeffs: np.ndarray) -> np.ndarray:
    out = np.zeros_like(bands["blue"], dtype="float32")
    for c, name in zip(coeffs, REQUIRED_BANDS):
        out = out + c * bands[name]
    return out


def compute_indices(bands: dict) -> dict:
    """`bands` is a dict of canonical band name -> 2D float array of
    surface reflectance (0-1 fraction, NaN where invalid/masked), with at
    least the six keys in REQUIRED_BANDS. Returns a dict metric name ->
    2D array (NDVI, TCB, TCG, TCW)."""
    missing = [b for b in REQUIRED_BANDS if b not in bands]
    if missing:
        raise ValueError(f"compute_indices: missing required band(s): {missing}")

    red, nir = bands["red"], bands["nir"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)

    return {
        "NDVI": ndvi.astype("float32"),
        "TCB": _tc(bands, TC_BRIGHTNESS),
        "TCG": _tc(bands, TC_GREENNESS),
        "TCW": _tc(bands, TC_WETNESS),
    }


def compute_ndsi(bands: dict) -> np.ndarray:
    """`bands` is a dict of canonical band name -> 2D float array of surface
    reflectance (0-1 fraction, NaN where invalid/masked), with at least
    "green" and "swir1". Returns NDSI = (green - swir1) / (green + swir1) as
    a 2D float32 array. Deliberately separate from `compute_indices()` -
    see this module's docstring for why NDSI needs its own validity mask
    rather than sharing the vegetation-index one."""
    missing = [b for b in NDSI_REQUIRED_BANDS if b not in bands]
    if missing:
        raise ValueError(f"compute_ndsi: missing required band(s): {missing}")

    green, swir1 = bands["green"], bands["swir1"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndsi = (green - swir1) / (green + swir1)
    return ndsi.astype("float32")
