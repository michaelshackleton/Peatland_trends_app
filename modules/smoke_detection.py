"""
Per-pixel temporal anomaly detection for smoke/haze that DEA's own oa_fmask
cloud mask misses - adapted from the same technique in this repo's
`../Estuary openings local app/modules/connectivity.py`
(build_clear_sky_reference / build_temporal_anomaly_mask), re-implemented
here rather than imported, since each app in this repo is meant to be
self-contained.

The idea: smoke and thin haze consistently brighten the blue band relative
to how that same pixel normally looks on a clear day, even when fmask
still calls the pixel "clear" (fmask is a per-scene classifier, not aware
of a location's own history). So instead of a fixed brightness threshold,
this builds a *per-pixel* "clear-sky reference" - the low percentile of
blue-band reflectance seen at that exact pixel across every other scene in
this polygon's own time series, using only observations fmask itself
already called clear/water - and flags a scene where a meaningful fraction
of pixels sit well above their own reference.

This only needs each scene's already-clipped blue band + valid-pixel mask
(tiny arrays, one per scene, for a small bog/fen polygon), which
`analysis.py` caches for every scene it fetches anyway (see its module
docstring) - no extra DEA requests are needed to run this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Blue-band brightness excess (raw DN, DEA nbart bands are reflectance x
# 10000) above a pixel's own clear-sky reference that counts as a
# "temporal anomaly". Same starting value as the Estuary openings app -
# ~400 DN is roughly a 0.04 reflectance jump. This is a starting point, not
# a validated figure, and genuinely benefits from checking flagged scenes
# against their true-colour view (tab 3's "view this scene" feature) for
# your specific sites.
DEFAULT_SMOKE_THRESHOLD_DN = 400.0
DEFAULT_REFERENCE_PERCENTILE = 10.0
DEFAULT_MIN_CLEAR_OBS = 3


@dataclass
class ClearSkyReference:
    reference: np.ndarray  # 2D, NaN where no usable history
    n_obs: np.ndarray  # 2D, how many clear/water observations contributed


def build_clear_sky_reference(
    blue_stack: np.ndarray,
    valid_stack: np.ndarray,
    percentile: float = DEFAULT_REFERENCE_PERCENTILE,
    min_clear_obs: int = DEFAULT_MIN_CLEAR_OBS,
) -> ClearSkyReference:
    """`blue_stack` is (n_scenes, y, x) raw blue-band DN for every scene of
    one sensor/product over one polygon, already on a common grid.
    `valid_stack` is the matching (n_scenes, y, x) boolean mask of pixels
    already considered usable (clear/water fmask, non-nodata - i.e. the
    same `valid` mask analysis.py computes per scene for its own
    statistics). Returns the low-percentile blue value seen at each pixel
    across its own valid observations - a rough "least atmosphere-
    contaminated" baseline for that exact location - NaN wherever fewer
    than `min_clear_obs` valid observations exist."""
    n_obs = valid_stack.sum(axis=0)
    masked = np.where(valid_stack, blue_stack.astype("float32"), np.nan)
    with np.errstate(all="ignore"):  # all-NaN slices (never-valid pixel) are expected
        reference = np.nanpercentile(masked, percentile, axis=0)
    reference = np.where(n_obs >= min_clear_obs, reference, np.nan)
    return ClearSkyReference(reference=reference.astype("float32"), n_obs=n_obs)


def scene_smoke_pct(
    blue: np.ndarray,
    valid: np.ndarray,
    reference: np.ndarray,
    total_px: int,
    threshold_dn: float = DEFAULT_SMOKE_THRESHOLD_DN,
) -> float:
    """Percentage of the polygon's pixels (relative to `total_px`, the same
    denominator pct_valid/snow_pct use) that are both currently valid and
    anomalously bright relative to their own clear-sky reference. Returns
    0.0 if there's no reference at all yet (e.g. this is one of the first
    few scenes for this polygon/sensor)."""
    if total_px <= 0:
        return 0.0
    has_reference = np.isfinite(reference)
    excess = blue.astype("float32") - reference
    flagged = valid & has_reference & (excess > threshold_dn)
    return 100.0 * float(flagged.sum()) / total_px
