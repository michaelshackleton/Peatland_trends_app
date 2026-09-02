"""
Core analysis pipeline: for one selected polygon, fetches Landsat and/or
Sentinel-2 surface reflectance scenes from Digital Earth Australia's STAC
catalogue and computes per-pixel NDVI / Tasselled Cap Brightness /
Greenness / Wetness / NDSI (snow index) for each scene.

Two-stage design:

1. `run_analysis()` is the only part that talks to DEA. It fetches scenes
   in small batches (`batch_size` at a time) so it never holds more than
   one batch's worth of raw satellite rasters in memory - this is what
   keeps it safe to run on a server with limited RAM even over the full
   Landsat archive back to 1984. Every scene gets a row of summary
   statistics (means, snow/cloud/smoke %) in `AnalysisResult.scene_df` -
   that's what the time-series and phenology plots and all the quality
   filters actually work from, so none of them need per-pixel data to be
   kept around afterward.

   The one thing that *does* need per-pixel data after the fetch is the
   pixel-mean raster (a spatial map, not just a summary number), so each
   scene's small clipped metric arrays are kept in
   `AnalysisResult.scene_cache` for that. Everything else a scene's raw
   pixels were needed for - the smoke/haze clear-sky reference in
   particular, which briefly needs every scene's blue band at once - is
   computed once, right here, before those raw arrays are dropped; only
   the finished `smoke_pct` number (in `scene_df`) and the metric arrays
   (in `scene_cache`, for the raster) survive past this function.

2. Everything that used to be a "run-time" choice baked into step 1 -
   minimum valid-pixel %, maximum snow/cloud/smoke %, and now also manually
   excluded scenes - is instead applied *afterwards*, instantly, by
   `recompute_pixel_rasters()` / `filter_scene_df()`. `filter_scene_df()`
   works purely from `scene_df` (no raster data needed at all).
   `recompute_pixel_rasters()` works from `scene_cache`'s metric arrays.
   No second DEA request is needed to change a quality threshold or
   exclude a scene; the app's results tab calls these on every rerun so
   the raster/time-series/phenology plots stay live as the user adjusts
   filters.

   The trade-off: memory still scales with (number of scenes) x (pixels
   per scene) for the lifetime of one run, but only for the metric arrays
   the raster needs - not the raw bands, fmask, or per-pixel masks those
   scenes were fetched with, which are all discarded once their scene's
   summary row and (if relevant) smoke reference contribution are done
   with them. For the polygon sizes this app deals with (bog/fen patches,
   at most a few hundred hectares) that's negligible - a few MB even for a
   thousand-scene run - but it's worth knowing if this is ever pointed at
   a much larger polygon, where it can grow into the hundreds of MB (see
   README's memory section).

A single scene's full-resolution rasters (for the "view this scene on a
map" feature) are never cached at all - `fetch_single_scene()` re-fetches
just that one scene from DEA on demand, and the app only ever holds one
such scene in memory at a time (see app.py's tab 3).

Reuses the vendored `STACDataManager` / `RSDataProductManager` classes from
`vendor/rs_data.py` / `vendor/rs_processing.py` (same DEA access pattern as
the sibling `Satellite downloader` app).

Smoke/haze detection (`smoke_pct` below) is a per-pixel temporal-anomaly
check adapted from `../Estuary openings local app/modules/connectivity.py`
- see `modules/smoke_detection.py`'s docstring.
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime as _dt
from typing import Callable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box

_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_MODULES_DIR)
_VENDOR_DIR = os.path.join(_APP_DIR, "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from rs_data import STACDataManager  # noqa: E402
from rs_processing import RSDataProductManager  # noqa: E402

from .indices import compute_indices, compute_ndsi, METRICS  # noqa: E402
from .smoke_detection import (  # noqa: E402
    DEFAULT_SMOKE_THRESHOLD_DN,
    build_clear_sky_reference,
    scene_smoke_pct,
)

# Neither `requests` nor `pystac_client` (which the STAC *search* step goes
# through - see vendor/rs_data.py's get_stac_items) set a network timeout by
# default. That means a connection that was mid-request when the internet
# drops can block indefinitely waiting for a response that will never come -
# and since the retry logic below (and fetch_single_scene()'s single-scene
# fetch) only ever gets a chance to run *after* a request raises an
# exception, an unbounded hang never reaches it; it just sits there. Setting
# a process-wide default socket timeout turns that silent, indefinite hang
# into a normal, raised TimeoutError after STAC_SOCKET_TIMEOUT_S seconds -
# generous enough for DEA's STAC gateway's normal latency, but bounded, so
# a genuinely dead connection is detected and retried (or reported) instead
# of leaving the app looking stuck with no way to tell. This only affects
# Python's own `socket` module (i.e. the STAC search calls); GDAL/rasterio's
# actual raster reads use their own libcurl networking stack, unaffected by
# this - that path is defended separately by BATCH_FETCH_MAX_ATTEMPTS's
# retry loop instead (see run_analysis()'s main loop).
STAC_SOCKET_TIMEOUT_S = 60.0
socket.setdefaulttimeout(STAC_SOCKET_TIMEOUT_S)

WGS84 = "EPSG:4326"
REFLECTANCE_SCALE = 10000.0
# DEA oa_fmask classes: 0 nodata, 1 clear, 2 cloud, 3 cloud shadow, 4 snow,
# 5 water. Only clear/water pixels are treated as usable for index
# calculations - snow and cloud/shadow are deliberately excluded (unlike
# rs_data.qa_daily_fmask_eval's own convention, which lumps snow in with
# "valid"): their reflectance is nothing like bare ground or vegetation, so
# averaging either in would badly skew NDVI/tasselled-cap means. Each is
# tracked separately per scene (snow_pct, cloud_pct columns) so it can be
# reported and optionally used to reject whole scenes, rather than
# silently contaminating the pixel statistics.
FMASK_VALID_CLASSES = [1, 5]
SNOW_FMASK_CLASS = 4
CLOUD_FMASK_CLASSES = [2, 3]
NODATA_FMASK_CLASS = 0

# NDSI needs a *broader* validity mask than the vegetation indices above:
# the whole point of adding it is to reveal snow-covered scenes, so its mask
# must keep the snow class rather than exclude it - it only drops nodata and
# cloud/shadow, where reflectance isn't meaningful for any index.
NDSI_VALID_FMASK_CLASSES = [1, 4, 5]

# Standard MODIS-convention NDSI snow threshold (Hall et al. 1995,
# Riggs/Hall/Salomonson MOD10 algorithm): pixels with NDSI above this are
# classed as snow. Exposed as a fetch-time parameter (tab 2's "Advanced
# fetch options"), like smoke_threshold_dn, since it's baked into
# scene_df's snow_pct_ndsi at fetch time rather than recomputed live.
DEFAULT_NDSI_SNOW_THRESHOLD = 0.4

# tab 3's "snow detection method" choice - which of the three per-scene
# snow_pct estimates the "maximum scene snow cover" quality filter checks
# against. "mixture" is a simple average of the fmask- and NDSI-based
# percentages: neither detector is perfect (fmask can miss thin/patchy snow
# fmask's algorithm wasn't tuned to catch; NDSI can be fooled by bright
# non-snow surfaces or shadowed snow), so averaging is a reasonable,
# easy-to-explain default rather than picking a winner.
DEFAULT_SNOW_METHOD = "fmask"
SNOW_METHOD_LABELS = {
    "fmask": "Cloud mask (oa_fmask snow class)",
    "ndsi": "NDSI (green/SWIR index)",
    "mixture": "Mixture (average of both)",
}

# The single biggest driver of this app's memory footprint is
# scene_cache: every fetched scene's full-resolution per-pixel metric
# arrays, kept for the whole session so tab 3's quality filters/scene
# exclusion can recompute the pixel-mean raster instantly without
# re-fetching. On Streamlit Community Cloud every visitor shares one
# process and one memory budget, so a handful of concurrent sessions each
# holding a large polygon's full-resolution cache can exhaust it. Block-
# mean downsampling what's stored (not what's used for quality-filter
# stats, which stay full-resolution - see run_analysis()) trades spatial
# detail in the pixel-mean raster map for a quadratic reduction in cache
# size (factor=2 -> 1/4 the pixels, factor=4 -> 1/16). 1 disables it.
DEFAULT_RASTER_DOWNSAMPLE_FACTOR = 2

ProgressCB = Optional[Callable[[str, float], None]]

# DEA's STAC gateway occasionally returns a 502/503/504 (or a plain
# connection/timeout error) under load - a transient server-side blip, not
# anything wrong with the request itself. `_search_stac_items` below splits
# a search into a few-year chunks and retries each one on exactly this kind
# of error, so a single flaky request only costs a retry of a few years'
# worth of search rather than losing an entire decade-long fetch (see its
# docstring, and run_analysis()'s for how this fits into the fetch step).
STAC_SEARCH_CHUNK_YEARS = 3
STAC_SEARCH_MAX_ATTEMPTS = 3
STAC_SEARCH_RETRY_BACKOFF_S = 3.0

# The STAC *search* step above already retries on a transient error, but
# the actual per-batch raster fetch in run_analysis()'s main loop
# (stac_items_to_xrdataset / make_product_dask - reading the pixel data
# itself from DEA's S3-backed store) did not, until a run on Streamlit
# Community Cloud lost roughly 15 years of Landsat data to a run of
# consecutive "Could not resolve host" (DNS) failures: since batches are
# chronological slices of a sorted item list, a stretch of consecutive
# batch failures wipes out a contiguous stretch of the timeline, not just
# scattered individual scenes - so this needed the same retry treatment as
# the search step, not a smaller one.
BATCH_FETCH_MAX_ATTEMPTS = 3
BATCH_FETCH_RETRY_BACKOFF_S = 5.0

# `filter_stac_items_region_overlap` (vendor/rs_data.py) computes each
# candidate scene's coverage as (intersection area of the STAC item's
# footprint polygon with the ROI) / (ROI area), then keeps the scene only if
# that ratio is >= the requested threshold. When a user asks for "100%"
# tile coverage (tab 2's slider defaults to exactly this), a scene that
# genuinely fully contains the polygon can still come out as 0.999999...
# rather than a clean 1.0 - ordinary floating-point noise from reprojecting
# both geometries and computing a polygon intersection, not a real gap in
# coverage. Comparing that against a literal 1.0 threshold then silently
# rejects every scene, including ones that visibly, entirely cover the
# polygon - which is exactly what "100%" was supposed to mean. Capping the
# threshold actually passed to the filter at just under 1.0 fixes this
# without weakening the filter for any other value: a request for 100% now
# behaves as "fully covers, allowing for floating-point noise" instead of
# "mathematically perfect containment", while lower requests (50%, 90%, ...)
# are unaffected since min() only ever pulls values back down when they'd
# have been >= this cap. The sibling `Estuary openings app` hit the same
# issue and uses the identical fix (its FULL_COVERAGE_THRESHOLD = 0.999).
FULL_COVERAGE_EPSILON = 0.001


def _is_transient_network_error(exc: Exception) -> bool:
    """True for errors worth retrying (a network/DNS/server-side hiccup,
    not a genuine problem with the request itself) rather than giving up
    immediately. Used both for the STAC *search* retries below and for
    retrying a batch's actual raster fetch (see run_analysis's main loop) -
    DNS resolution failures ("Could not resolve host") are included
    because they've been observed on Streamlit Community Cloud as a brief,
    self-resolving DNS-resolver hiccup reaching DEA's S3-backed data
    endpoint, not a real, lasting network outage. This matters most for the
    per-batch fetch: unlike the STAC search step (already chunked and
    retried), a batch's raster fetch previously had no retry at all, so a
    short DNS blip there would silently drop every scene in every batch it
    touched for the rest of a run - potentially years of data, since
    batches are chronological slices of a sorted item list, so a run of
    consecutive failures wipes out a contiguous stretch of the timeline.
    Also matches the plain `TimeoutError`/"timed out" that
    `socket.setdefaulttimeout()` (see STAC_SOCKET_TIMEOUT_S above) now
    raises for a connection that hangs rather than erroring outright -
    case-insensitive, since exact wording/casing varies across urllib3,
    requests, and the underlying socket/curl layers."""
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "502", "503", "504", "timeout", "timed out", "connection",
            "could not resolve host", "name or service not known",
            "temporary failure in name resolution", "network is unreachable",
            "curl error",
        )
    )


def _date_chunks(start_date: str, end_date: str, chunk_years: int) -> list[tuple[str, str]]:
    """Splits an ISO start/end date pair into a list of (chunk_start,
    chunk_end) ISO date string pairs, each spanning at most `chunk_years`
    years, covering the full range in order with no gaps or overlaps."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []
    cur = start
    while cur <= end:
        nxt = min(cur + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1), end)
        chunks.append((cur.date().isoformat(), nxt.date().isoformat()))
        cur = nxt + pd.Timedelta(days=1)
    return chunks


def _search_stac_items(
    stac_mgr, product_mgr, region_mgr, start_date: str, end_date: str, sensor_label: str,
    report: Callable[[str, float], None], frac_start: float, frac_end: float, warnings: list,
) -> list:
    """Searches DEA's STAC catalogue for one product across the full date
    range, a few years at a time (STAC_SEARCH_CHUNK_YEARS), retrying each
    chunk up to STAC_SEARCH_MAX_ATTEMPTS times on a transient server error
    with a short backoff between attempts. A chunk that still fails after
    every attempt is skipped (with a warning appended describing exactly
    which period is missing and why) rather than aborting the whole
    product's search - a multi-year fetch shouldn't lose everything it
    already found just because one gateway request timed out partway
    through. Returns the combined item list across every chunk that
    succeeded (possibly all of them, possibly none)."""
    chunks = _date_chunks(start_date, end_date, STAC_SEARCH_CHUNK_YEARS)
    all_items = []
    n_chunks = max(len(chunks), 1)
    for i, (c_start, c_end) in enumerate(chunks):
        frac = frac_start + (frac_end - frac_start) * (i / n_chunks)
        report(f"Searching {sensor_label} scenes {c_start} to {c_end}...", frac)
        last_exc = None
        for attempt in range(STAC_SEARCH_MAX_ATTEMPTS):
            try:
                all_items.extend(stac_mgr.get_stac_items(c_start, c_end, product_mgr, region_mgr))
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < STAC_SEARCH_MAX_ATTEMPTS - 1 and _is_transient_network_error(e):
                    time.sleep(STAC_SEARCH_RETRY_BACKOFF_S * (attempt + 1))
                    continue
                break
        if last_exc is not None:
            warnings.append(
                f"{sensor_label}: STAC search failed for {c_start} to {c_end} after "
                f"{STAC_SEARCH_MAX_ATTEMPTS} attempt(s) - {last_exc}. Scenes from this "
                "period will be missing from this run; try running again (a gateway "
                "timeout is usually transient), or narrow the date range."
            )
    return all_items


def auto_utm_epsg(lon: float, lat: float) -> int:
    """WGS84 UTM zone EPSG code for a given lon/lat (southern hemisphere
    for anywhere in Australia)."""
    zone = int((lon + 180) // 6) + 1
    return (32700 if lat < 0 else 32600) + zone


class _SinglePolygonRegionManager:
    """Duck-types the subset of STACRSRegionManager's interface that
    STACDataManager / RSDataProductManager actually use, backed by a single
    in-memory polygon rather than a shapefile on disk."""

    def __init__(self, roi_gdf: gpd.GeoDataFrame, region_name: str):
        self.region_poly = roi_gdf
        self.region_name = region_name
        self.region_code = region_name
        self.sub_region_col = None

    def get_polygon(self, sub_region=None, target_crs=None):
        gdf = self.region_poly.to_crs(target_crs) if target_crs else self.region_poly
        return gdf.iloc[0].geometry

    def get_region_extent(self, sub_region=None, target_crs=None):
        gdf = self.region_poly.to_crs(target_crs) if target_crs else self.region_poly
        return gdf.total_bounds


def _polygon_pixel_mask(roi_gdf_projected: gpd.GeoDataFrame, ref_da) -> np.ndarray:
    """Rasterises the polygon onto `ref_da`'s grid with all_touched=True -
    i.e. any pixel the polygon boundary touches counts as "in polygon", not
    just pixels whose centre falls inside it. This matters a lot here: many
    of these bog/fen polygons are well under one Landsat pixel (0.09 ha) in
    area, so a centre-in-polygon rasterisation would silently return zero
    pixels for a lot of them."""
    shape2d = ref_da.shape[-2:]
    transform = ref_da.rio.transform()
    shapes = [(geom, 1) for geom in roi_gdf_projected.geometry]
    mask = rasterize(
        shapes=shapes,
        out_shape=shape2d,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return mask == 1


def _downsample_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downsample a 2D array by `factor` (e.g. 2 halves each
    dimension, so the array has 1/4 the pixels) - NaN-aware, so a block is
    NaN in the output only if every pixel in it is NaN. Any trailing
    row/column that doesn't fill a whole block is dropped, and the array
    is returned unchanged if it's too small in either dimension for even
    one full block at this factor (rather than collapsing a tiny polygon
    to a 0-pixel array). Pair with `_downsample_transform()` so the
    returned array's geolocation stays correct."""
    if factor <= 1:
        return arr
    h, w = arr.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    if h2 < factor or w2 < factor:
        return arr
    cropped = arr[:h2, :w2]
    reshaped = cropped.reshape(h2 // factor, factor, w2 // factor, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(reshaped, axis=(1, 3))


def _downsample_transform(transform, arr_shape: tuple, factor: int):
    """The affine transform for `_downsample_mean(arr, factor)`'s output -
    same top-left origin as `transform`, pixels `factor` times bigger. If
    the array was too small to actually downsample (see
    `_downsample_mean`'s guard), returns `transform` unchanged so the two
    stay consistent with each other."""
    if factor <= 1:
        return transform
    h, w = arr_shape
    if (h // factor) * factor < factor or (w // factor) * factor < factor:
        return transform
    return transform * Affine.scale(factor)


# fetch_single_scene()'s landscape-context buffer - see that function's
# docstring for why this is a square rather than just the polygon's own
# (often much smaller and more irregularly-shaped) bounding box padded out.
# SCENE_VIEW_BUFFER_M is the *floor*: most bog/fen polygons are tiny, so a
# flat 200m buffer around them would only show a few hundred metres of
# landscape context - not enough to orient the view. _scene_view_buffer_m()
# below instead aims for a SCENE_VIEW_TARGET_SIZE_M (2km) square by default,
# expanding the buffer as needed for small polygons, and only falls back to
# this flat minimum once the polygon itself is already big enough that a 2km
# target would under-buffer (or even crop) it.
SCENE_VIEW_BUFFER_M = 200.0
SCENE_VIEW_TARGET_SIZE_M = 2000.0


def _scene_view_buffer_m(
    roi_gdf: gpd.GeoDataFrame,
    target_crs: int,
    target_size_m: float = SCENE_VIEW_TARGET_SIZE_M,
    min_buffer_m: float = SCENE_VIEW_BUFFER_M,
) -> float:
    """Buffer distance (metres, applied on every side by
    `_buffered_square_gdf`) for the "Inspect scene" viewer: enough to bring
    the total square up to `target_size_m` (2km x 2km by default) for small
    polygons, but never less than `min_buffer_m`. For a polygon whose own
    extent already exceeds `target_size_m`, aiming for a flat 2km square
    would under-buffer it (or even crop pixels off), so the buffer falls
    back to the flat minimum around the polygon's own (already large)
    extent instead - i.e. exactly the old fixed-200m behaviour, but only
    once the polygon is big enough that "2km square" no longer makes sense
    as a target."""
    projected = roi_gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = projected.total_bounds
    extent = max(maxx - minx, maxy - miny)
    buffer_for_target = (target_size_m - extent) / 2.0
    return max(buffer_for_target, min_buffer_m)


def _buffered_square_gdf(roi_gdf: gpd.GeoDataFrame, target_crs: int, buffer_m: float) -> gpd.GeoDataFrame:
    """Returns a 1-row GeoDataFrame (in `roi_gdf`'s own CRS) whose geometry
    is a square bounding box around `roi_gdf`'s polygon, expanded by
    `buffer_m` on every side - computed in `target_crs` (a projected,
    metres-based CRS) so the buffer distance is a real distance rather than
    degrees. Square (not just the padded rectangle) so the fetched scene
    isn't lopsided around a polygon that's much longer in one direction
    than the other - the point is to show equal landscape context on every
    side, not just pad the existing shape."""
    projected = roi_gdf.to_crs(target_crs)
    minx, miny, maxx, maxy = projected.total_bounds
    minx, miny, maxx, maxy = minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    half_side = max(maxx - minx, maxy - miny) / 2.0
    square = box(cx - half_side, cy - half_side, cx + half_side, cy + half_side)
    square_gdf = gpd.GeoDataFrame(geometry=[square], crs=projected.crs)
    return square_gdf.to_crs(roi_gdf.crs)


def cache_size_bytes(result: "AnalysisResult") -> int:
    """Total memory footprint (bytes) of `result.scene_cache` as it stands
    by the time run_analysis() returns - in practice just each scene's
    metric arrays (valid_mask/polygon_mask/blue_dn are released once
    smoke_pct has been computed, see SceneCacheEntry's docstring), summed
    via numpy's own `.nbytes`. Used to give the user a real, measured
    number for the pixel-raster cache's memory footprint rather than just
    an estimate - see tab 2's summary panel and the README's memory
    section."""
    total = 0
    for entry in result.scene_cache.values():
        if entry.valid_mask is not None:
            total += entry.valid_mask.nbytes
        if entry.polygon_mask is not None:
            total += entry.polygon_mask.nbytes
        if entry.blue_dn is not None:
            total += entry.blue_dn.nbytes
        for arr in entry.metrics.values():
            if arr is not None:
                total += arr.nbytes
    return total


def scene_key(product_code: str, item_id: str) -> str:
    """Stable string key identifying one scene, used for the excluded-scene
    set / cache lookups - safe to put in a Streamlit multiselect or
    session_state set (unlike a tuple, which some Streamlit widgets balk
    at)."""
    return f"{product_code}::{item_id}"


@dataclass
class SceneCacheEntry:
    product_code: str
    item_id: str
    # valid_mask/blue_dn only exist transiently, for _fill_smoke_pct()'s
    # clear-sky reference (built once, right after fetching, from every
    # scene of a product at once). run_analysis() sets both back to None
    # once that's done and smoke_pct has been written into scene_df -
    # nothing needs them after that, so there's no reason to keep them
    # around for the rest of the session. polygon_mask isn't used by
    # anything downstream at all (recompute_pixel_rasters works out which
    # pixels are valid from the metric arrays' own NaNs) so it's never
    # even set - kept as a field only so old code/notebooks poking at a
    # SceneCacheEntry don't hit an AttributeError.
    valid_mask: Optional[np.ndarray]   # bool, pm & non-nodata bands & fmask clear/water
    polygon_mask: Optional[np.ndarray]  # unused downstream; always None
    metrics: dict             # metric name -> 2D float16 array (NaN outside valid_mask), possibly block-mean downsampled (see DEFAULT_RASTER_DOWNSAMPLE_FACTOR) - kept, needed by recompute_pixel_rasters
    blue_dn: Optional[np.ndarray]  # raw (unscaled) blue band, for smoke detection
    transform: object
    crs: object


@dataclass
class AnalysisResult:
    scene_df: pd.DataFrame
    scene_cache: dict          # scene_key -> SceneCacheEntry
    stac_items: dict           # product_code -> {item_id: pystac.Item} (for single-scene re-fetch)
    polygon_uid: str
    polygon_geom_wgs84: object
    target_crs: int
    polygon_area_ha: dict       # product_code -> polygon pixel-mask area in ha (diagnostic)
    polygon_total_px: dict      # product_code -> polygon pixel count on that product's grid
    products_json_path: str
    fetch_params: dict = field(default_factory=dict)  # what was asked for at fetch time, for display
    warnings: list = field(default_factory=list)
    # product_code -> boolean 2D array (True = inside the polygon) on that
    # product's grid - the same mask used to derive polygon_area_ha/
    # polygon_total_px above, but kept as the actual array (not just its
    # summary numbers) so tab 3's persistent-valid-fraction diagnostic can
    # clip its map to the polygon's own interior instead of being
    # dominated by the buffered background around it, which is NaN in
    # every scene simply because it's outside the polygon, not because of
    # any real cloud/shadow/footprint issue.
    polygon_masks: dict = field(default_factory=dict)


def run_analysis(
    polygon_geom_wgs84,
    polygon_uid: str,
    product_codes: list[str],
    start_date: str,
    end_date: str,
    products_json_path: str,
    max_cloud: float = 100.0,
    min_coverage_pct: float = 100.0,
    smoke_threshold_dn: float = DEFAULT_SMOKE_THRESHOLD_DN,
    ndsi_snow_threshold: float = DEFAULT_NDSI_SNOW_THRESHOLD,
    batch_size: int = 15,
    downsample_factor: int = DEFAULT_RASTER_DOWNSAMPLE_FACTOR,
    progress_cb: ProgressCB = None,
) -> AnalysisResult:
    """Fetches every scene from DEA (this is the only function in this
    module that makes network requests) and caches each scene's per-pixel
    arrays for later, instant, re-filtering.

    `max_cloud` (0-100) is a whole-scene cloud-cover ceiling applied at the
    STAC search stage (DEA's own eo:cloud_cover field, computed over the
    *entire tile* - often 100km+ across - not just this polygon, so a low
    setting can rule out scenes that are perfectly clear over the polygon
    itself; this is really just a fetch-volume limiter). `min_coverage_pct`
    (0-100) drops any scene whose tile footprint doesn't cover at least
    that percentage of the polygon - also applied at the STAC-search stage,
    before any raster is fetched. 100 doesn't require mathematically exact
    full containment (see FULL_COVERAGE_EPSILON) - a scene that visibly
    fully covers the polygon won't be rejected over floating-point noise in
    the overlap calculation.

    Per-polygon data-quality filtering (valid-pixel %, snow %, cloud % -
    this time computed over the polygon itself, not the whole tile - and
    smoke/haze %) is NOT done here; see `filter_scene_df()` and
    `recompute_pixel_rasters()` below, which work from this function's
    cached output without needing to re-fetch anything. The one exception
    is `ndsi_snow_threshold`: like `smoke_threshold_dn`, it's baked into
    scene_df's snow_pct_ndsi column at fetch time (not adjustable live),
    since it depends on a per-pixel NDSI array only kept transiently here.

    `downsample_factor` (see its module-level docstring) block-mean
    downsamples each scene's cached metric arrays before they're stored in
    `scene_cache`, to keep the session's memory footprint down - it does
    NOT affect n_valid_px/pct_valid/snow_pct/cloud_pct or any other
    per-scene statistic in scene_df, which are always computed from the
    full-resolution arrays first. Changing it for an already-fetched
    polygon means the cached arrays are at the wrong resolution, so the
    app clears and re-fetches rather than trying to reconcile the two -
    see tab 2's resolution control.

    The STAC *search* itself (finding which scenes exist, before fetching
    any of their pixels) is done a few years at a time and retried on
    transient server errors - see `_search_stac_items()` - since a long
    date range searched in one request is exactly the kind of thing DEA's
    STAC gateway occasionally 504s on, and losing an entire multi-decade
    search over one flaky request would be a bad trade for a retry that
    usually just works. Each batch's actual raster fetch is retried the
    same way (up to BATCH_FETCH_MAX_ATTEMPTS times) on the same class of
    transient error, including DNS resolution failures - see
    _is_transient_network_error()'s docstring for why this matters even
    more here than for the search step.
    """
    if not product_codes:
        raise ValueError("Select at least one of Landsat / Sentinel-2.")

    with open(products_json_path) as f:
        all_products = json.load(f)

    roi_gdf = gpd.GeoDataFrame(geometry=[polygon_geom_wgs84], crs=WGS84)
    centroid = roi_gdf.geometry.iloc[0].centroid
    target_crs = auto_utm_epsg(centroid.x, centroid.y)
    region_mgr = _SinglePolygonRegionManager(roi_gdf, region_name=polygon_uid)
    roi_projected = roi_gdf.to_crs(target_crs)

    work_dir = tempfile.mkdtemp(prefix="peatland_trends_")
    stac_mgr = STACDataManager(root_dir=work_dir)

    scene_rows: list[dict] = []
    scene_cache: dict = {}
    stac_items: dict = {}
    polygon_masks: dict = {}
    polygon_area_ha: dict = {}
    polygon_total_px: dict = {}
    warnings: list[str] = []

    def report(msg, frac):
        if progress_cb:
            progress_cb(msg, max(0.0, min(1.0, frac)))

    n_products = len(product_codes)
    for p_idx, product_code in enumerate(product_codes):
        params = all_products[product_code]
        canonical_bands = params["canonical_bands"]
        sensor_label = params.get("sensor_label", product_code)
        nodata_val = params["nodata"]
        product_mgr = RSDataProductManager(product_code, product_param_path=products_json_path)

        items = _search_stac_items(
            stac_mgr, product_mgr, region_mgr, start_date, end_date, sensor_label,
            report, p_idx / n_products, (p_idx + 0.5) / n_products, warnings,
        )
        try:
            items = stac_mgr.filter_stac_items_eocloud(items, max_cloud)
            if min_coverage_pct > 0 and items:
                # See FULL_COVERAGE_EPSILON's docstring above - this keeps a
                # "100%" request from rejecting genuinely fully-covering
                # scenes over floating-point noise in the overlap calc.
                min_overlap_frac = min(min_coverage_pct / 100.0, 1.0 - FULL_COVERAGE_EPSILON)
                items = stac_mgr.filter_stac_items_region_overlap(
                    items, min_overlap_frac, region_mgr, target_crs=target_crs
                )
        except Exception as e:
            warnings.append(f"{sensor_label}: could not apply cloud/coverage filters to search results - {e}")
            continue
        items = sorted(items, key=lambda it: it.properties["datetime"])
        stac_items[product_code] = {item.id: item for item in items}

        n_items = len(items)
        if n_items == 0:
            warnings.append(
                f"{sensor_label}: no scenes found for this polygon/date range/cloud/coverage filter."
            )
            continue
        n_batches = math.ceil(n_items / batch_size)

        for b in range(n_batches):
            batch_items = items[b * batch_size:(b + 1) * batch_size]
            frac = (p_idx + (b / n_batches)) / n_products
            report(f"{sensor_label}: batch {b + 1}/{n_batches} ({len(batch_items)} scenes)...", frac)

            # Retried up to BATCH_FETCH_MAX_ATTEMPTS times on a transient
            # network/DNS error (see _is_transient_network_error) before
            # this batch is given up on - a short DNS/connection blip here
            # would otherwise silently drop every scene in this batch, and
            # since batches are chronological slices of a sorted item list,
            # a run of consecutive failures wipes out a contiguous stretch
            # of the timeline rather than scattered individual scenes.
            last_batch_exc = None
            for attempt in range(BATCH_FETCH_MAX_ATTEMPTS):
                try:
                    xr_ds = stac_mgr.stac_items_to_xrdataset(
                        batch_items, product_mgr, region_mgr, target_crs=target_crs
                    )
                    processed = product_mgr.make_product_dask(xr_ds, region_mgr, compute=True)
                    last_batch_exc = None
                    break
                except Exception as e:
                    last_batch_exc = e
                    if attempt < BATCH_FETCH_MAX_ATTEMPTS - 1 and _is_transient_network_error(e):
                        report(
                            f"{sensor_label}: batch {b + 1}/{n_batches} hit a transient error, "
                            f"retrying ({attempt + 1}/{BATCH_FETCH_MAX_ATTEMPTS})...", frac,
                        )
                        time.sleep(BATCH_FETCH_RETRY_BACKOFF_S * (attempt + 1))
                        continue
                    break
            if last_batch_exc is not None:
                warnings.append(
                    f"{sensor_label} batch {b + 1}/{n_batches}: failed to fetch/process after "
                    f"{BATCH_FETCH_MAX_ATTEMPTS} attempt(s) - {last_batch_exc}. Scenes from this "
                    "batch will be missing from this run; try running again (a network/DNS blip "
                    "is usually transient)."
                )
                continue

            if "time" not in processed.dims:
                processed = processed.expand_dims("time")

            ref_band_name = canonical_bands["red"]
            if ref_band_name not in processed.data_vars:
                warnings.append(f"{sensor_label} batch {b + 1}/{n_batches}: missing '{ref_band_name}' band, skipped.")
                continue

            if product_code not in polygon_masks:
                ref_slice = processed[ref_band_name].isel(time=0)
                try:
                    pm = _polygon_pixel_mask(roi_projected, ref_slice)
                except Exception as e:
                    warnings.append(f"{sensor_label}: could not rasterise polygon onto grid - {e}")
                    continue
                polygon_masks[product_code] = pm
                px_area_m2 = abs(ref_slice.rio.transform().a * ref_slice.rio.transform().e)
                polygon_area_ha[product_code] = float(pm.sum()) * px_area_m2 / 10000.0
                polygon_total_px[product_code] = int(pm.sum())

            pm = polygon_masks[product_code]
            if pm.shape != processed[ref_band_name].isel(time=0).shape:
                warnings.append(
                    f"{sensor_label} batch {b + 1}/{n_batches}: grid shape changed mid-run, batch skipped."
                )
                continue
            total_px = polygon_total_px[product_code]
            transform = processed.rio.transform()
            crs = processed.rio.crs
            # The transform that actually describes what gets stored in
            # scene_cache below (possibly downsampled) - see
            # _downsample_transform()'s docstring.
            cache_transform = _downsample_transform(transform, pm.shape, downsample_factor)

            # Map each STAC item's datetime (second precision) to its
            # metadata, so per-scene rows can be matched back up after
            # odc-stac turns the item list into time slices (mirrors the
            # same pattern used in Satellite downloader/modules/fetch.py).
            item_meta = {}
            for item in batch_items:
                try:
                    dt = _dt.fromisoformat(item.properties["datetime"].replace("Z", "+00:00"))
                    key = dt.strftime("%Y-%m-%dT%H:%M:%S")
                    item_meta[key] = dict(
                        item_id=item.id,
                        cloud_cover_pct=item.properties.get("eo:cloud_cover"),
                    )
                except Exception:
                    continue

            fmask_var = canonical_bands.get("fmask")
            time_values = np.atleast_1d(processed.time.values)

            for i, t in enumerate(time_values):
                scene = processed.isel(time=i)

                raw_bands = {}
                ok = True
                for canon, varname in canonical_bands.items():
                    if canon == "fmask":
                        continue
                    if varname not in scene.data_vars:
                        ok = False
                        break
                    raw_bands[canon] = np.asarray(scene[varname].values, dtype="float64")
                if not ok:
                    continue

                base_valid = pm.copy()
                for arr in raw_bands.values():
                    base_valid = base_valid & (arr != nodata_val) & np.isfinite(arr)
                snow_px = 0
                cloud_px = 0
                fmask_arr = None
                # `valid`: the vegetation-index gate (clear/water only, per
                # FMASK_VALID_CLASSES) - unchanged from before. `valid_ndsi`:
                # a broader gate that keeps the snow class too (per
                # NDSI_VALID_FMASK_CLASSES), so NDSI can still be computed -
                # and reveal snow cover - for scenes `valid` hard-excludes.
                if fmask_var and fmask_var in scene.data_vars:
                    fmask_arr = np.asarray(scene[fmask_var].values)
                    valid = base_valid & np.isin(fmask_arr, FMASK_VALID_CLASSES)
                    valid_ndsi = base_valid & np.isin(fmask_arr, NDSI_VALID_FMASK_CLASSES)
                    snow_px = int((pm & (fmask_arr == SNOW_FMASK_CLASS)).sum())
                    cloud_px = int((pm & np.isin(fmask_arr, CLOUD_FMASK_CLASSES)).sum())
                else:
                    valid = base_valid
                    valid_ndsi = base_valid

                valid_px = int(valid.sum())
                valid_px_ndsi = int(valid_ndsi.sum())
                pct_valid = (100.0 * valid_px / total_px) if total_px else 0.0
                pct_valid_ndsi = (100.0 * valid_px_ndsi / total_px) if total_px else 0.0
                snow_pct_fmask = (100.0 * snow_px / total_px) if total_px else 0.0
                cloud_pct = (100.0 * cloud_px / total_px) if total_px else 0.0

                metrics = {}
                if valid_px > 0:
                    refl = {
                        canon: np.where(valid, arr / REFLECTANCE_SCALE, np.nan)
                        for canon, arr in raw_bands.items()
                    }
                    metrics = compute_indices(refl)

                snow_px_ndsi = 0
                if valid_px_ndsi > 0 and "green" in raw_bands and "swir1" in raw_bands:
                    refl_ndsi = {
                        "green": np.where(valid_ndsi, raw_bands["green"] / REFLECTANCE_SCALE, np.nan),
                        "swir1": np.where(valid_ndsi, raw_bands["swir1"] / REFLECTANCE_SCALE, np.nan),
                    }
                    ndsi_arr = compute_ndsi(refl_ndsi)
                    metrics["NDSI"] = ndsi_arr
                    snow_px_ndsi = int(np.nansum(ndsi_arr > ndsi_snow_threshold))
                snow_pct_ndsi = (100.0 * snow_px_ndsi / total_px) if total_px else 0.0

                ts = pd.Timestamp(t)
                meta = item_meta.get(ts.strftime("%Y-%m-%dT%H:%M:%S"), {})
                item_id = meta.get("item_id") or f"{product_code}_{ts.isoformat()}"
                row = dict(
                    datetime=ts,
                    date=ts.date().isoformat(),
                    product=product_code,
                    sensor=sensor_label,
                    item_id=item_id,
                    cloud_cover_pct=meta.get("cloud_cover_pct"),
                    n_valid_px=valid_px,
                    n_valid_px_ndsi=valid_px_ndsi,
                    n_polygon_px=total_px,
                    pct_valid=round(pct_valid, 1),
                    pct_valid_ndsi=round(pct_valid_ndsi, 1),
                    # snow_pct is kept as an alias of snow_pct_fmask (the
                    # original, sole snow estimate) for backward
                    # compatibility with anything reading an older-shaped
                    # scene_df (e.g. tab 4 CSV exports from before NDSI was
                    # added). See SNOW_METHOD_LABELS / _snow_pct_for_method
                    # for how tab 3 actually picks between the three.
                    snow_pct=round(snow_pct_fmask, 1),
                    snow_pct_fmask=round(snow_pct_fmask, 1),
                    snow_pct_ndsi=round(snow_pct_ndsi, 1),
                    cloud_pct=round(cloud_pct, 1),
                    smoke_pct=0.0,  # filled in below, once every scene's been fetched
                )
                for m in METRICS:
                    arr = metrics.get(m)
                    # NDSI is gated on its own (broader) valid-pixel count,
                    # not the vegetation-index one - see this loop's masks
                    # above for why they can differ for the same scene.
                    m_valid_px = valid_px_ndsi if m == "NDSI" else valid_px
                    if arr is not None and m_valid_px > 0:
                        row[f"{m}_mean"] = float(np.nanmean(arr))
                        row[f"{m}_std"] = float(np.nanstd(arr))
                    else:
                        row[f"{m}_mean"] = np.nan
                        row[f"{m}_std"] = np.nan
                scene_rows.append(row)

                if valid_px > 0 or valid_px_ndsi > 0:
                    # row[f"{m}_mean"/"_std"] above are already computed from
                    # the full-resolution arrays, so it's safe to shrink
                    # `metrics` now, for what actually gets kept in
                    # scene_cache for the rest of the session - block-mean
                    # downsampled (see DEFAULT_RASTER_DOWNSAMPLE_FACTOR) and
                    # stored as float16 rather than float64/float32, since
                    # these are index values in roughly [-1, 1] that don't
                    # need more precision than that for the pixel-mean
                    # raster map they're used for.
                    cached_metrics = {
                        m: _downsample_mean(arr, downsample_factor).astype("float16")
                        for m, arr in metrics.items()
                        if arr is not None
                    }
                    scene_cache[scene_key(product_code, item_id)] = SceneCacheEntry(
                        product_code=product_code,
                        item_id=item_id,
                        valid_mask=valid,        # released below, once smoke_pct is filled in
                        polygon_mask=None,       # never used downstream - not worth keeping
                        metrics=cached_metrics,
                        blue_dn=raw_bands.get("blue"),  # released below, once smoke_pct is filled in
                        transform=cache_transform,
                        crs=crs,
                    )

            del xr_ds, processed

    report("Checking for smoke/haze not flagged by cloud masking...", 0.97)
    scene_df = pd.DataFrame(scene_rows)
    if len(scene_df) > 0:
        scene_df = scene_df.sort_values("datetime").reset_index(drop=True)
        _fill_smoke_pct(scene_df, scene_cache, product_codes, polygon_total_px, smoke_threshold_dn)

    # smoke_pct is now a finished number in scene_df for every scene - the
    # raw blue-band/valid-mask arrays that went into computing it were only
    # ever needed for that one calculation (see SceneCacheEntry's
    # docstring), so release them now rather than holding them in memory
    # for the rest of the session. Only the metric arrays (needed by
    # recompute_pixel_rasters, for the pixel-mean raster) survive past
    # this point.
    for entry in scene_cache.values():
        entry.valid_mask = None
        entry.blue_dn = None

    report("Done.", 1.0)

    return AnalysisResult(
        scene_df=scene_df,
        scene_cache=scene_cache,
        stac_items=stac_items,
        polygon_uid=polygon_uid,
        polygon_geom_wgs84=polygon_geom_wgs84,
        target_crs=target_crs,
        polygon_area_ha=polygon_area_ha,
        polygon_total_px=polygon_total_px,
        polygon_masks=polygon_masks,
        products_json_path=products_json_path,
        fetch_params=dict(
            start_date=start_date, end_date=end_date, product_codes=list(product_codes),
            max_cloud=max_cloud, min_coverage_pct=min_coverage_pct,
            smoke_threshold_dn=smoke_threshold_dn, ndsi_snow_threshold=ndsi_snow_threshold,
            batch_size=batch_size, downsample_factor=downsample_factor,
        ),
        warnings=warnings,
    )


def _fill_smoke_pct(scene_df, scene_cache, product_codes, polygon_total_px, threshold_dn):
    """Second pass, purely from the cache built by run_analysis (no DEA
    request): builds each product's per-pixel clear-sky blue-band
    reference from every cached scene, then fills in scene_df's smoke_pct
    column. Mutates scene_df in place."""
    for product_code in product_codes:
        entries = [
            e for k, e in scene_cache.items()
            if e.product_code == product_code and e.blue_dn is not None
        ]
        if len(entries) < 3:
            continue  # not enough of a time series yet to build a meaningful reference
        blue_stack = np.stack([e.blue_dn for e in entries])
        valid_stack = np.stack([e.valid_mask for e in entries])
        ref = build_clear_sky_reference(blue_stack, valid_stack)
        total_px = polygon_total_px.get(product_code, 0)
        for e in entries:
            pct = scene_smoke_pct(e.blue_dn, e.valid_mask, ref.reference, total_px, threshold_dn=threshold_dn)
            mask = (scene_df["product"] == product_code) & (scene_df["item_id"] == e.item_id)
            scene_df.loc[mask, "smoke_pct"] = round(pct, 1)


def _snow_pct_for_method(df: pd.DataFrame, snow_method: str) -> pd.Series:
    """Picks (or computes) the per-scene snow_pct Series to check the
    "maximum scene snow cover" filter against, per `snow_method` ("fmask" /
    "ndsi" / "mixture" - see SNOW_METHOD_LABELS). Falls back to the plain
    `snow_pct` column (present even in pre-NDSI scene_df's, e.g. an older
    tab 4 CSV export) wherever a method-specific column is missing, so this
    degrades gracefully rather than raising."""
    method = (snow_method or DEFAULT_SNOW_METHOD).lower()
    legacy = df["snow_pct"] if "snow_pct" in df.columns else pd.Series(0.0, index=df.index)
    fmask_col = df["snow_pct_fmask"] if "snow_pct_fmask" in df.columns else legacy
    if method == "ndsi":
        return df["snow_pct_ndsi"] if "snow_pct_ndsi" in df.columns else legacy
    if method == "mixture":
        if "snow_pct_ndsi" in df.columns:
            return (fmask_col + df["snow_pct_ndsi"]) / 2.0
        return fmask_col
    return fmask_col  # "fmask" (default)


# Default snow-cover thresholds (%) for the melt-out date table/plot in tab
# 3's "Snow cover & melt-out dates" section - a full ladder from fully
# snow-covered down to bare, at a coarse enough step to be readable as both
# a table and a set of reference markers on the seasonal plot.
DEFAULT_MELT_THRESHOLDS = list(range(100, -1, -10))

# "Melted out" for the headline single-date summary is defined as snow_pct
# dropping to (or below) this value, not literally 0 - a handful of
# NDSI-misclassified pixels (shadow, bright rock, a lingering wet patch)
# can keep a genuinely bare polygon's snow_pct hovering just above zero for
# the rest of the season, which would otherwise make "0%" never trigger, or
# trigger absurdly late waiting for a perfectly clean scene. 5% is a
# starting point, not a validated figure for these specific sites - exposed
# as a tab 3 control so it can be tuned per polygon.
DEFAULT_MELT_OUT_THRESHOLD_PCT = 5.0

MELT_STATUS_LABELS = {
    "crossed": "Crossed (interpolated date below)",
    "not_reached": "Never reached this threshold",
    "still_covered": "Still at/above this threshold at last observation",
    "insufficient_data": "Fewer than 2 usable scenes this season",
}


def snow_threshold_crossings(
    dates,
    snow_pct,
    thresholds: Optional[list] = None,
    mode: str = "first",
) -> pd.DataFrame:
    """For one snow season's worth of (date, snow_pct) observations -
    already restricted to a single melt season/year by the caller, e.g. one
    calendar year's scenes filtered to the months snow can plausibly be
    present (tab 3's "Snow season months" control) - finds, for each value
    in `thresholds` (default DEFAULT_MELT_THRESHOLDS, 100 -> 0 in steps of
    10), a date the polygon's snow % drops from at or above that threshold
    to below it, linearly interpolating between the two bracketing
    observations' dates in proportion to how far the true crossing point
    sits between their snow_pct values. This is the core calculation
    behind tab 3's "Snow cover & melt-out dates" section.

    Real time series aren't perfectly monotonic - a late cold snap can
    push snow_pct back up mid-melt, briefly re-crossing a threshold
    upward before crossing back down again later - so `mode` controls
    which downward crossing is reported when a threshold is crossed more
    than once:
    - "first" (default): the earliest downward crossing - useful for
      "when did it first start dropping from this level", but can flag an
      early, temporary partial-melt dip that later re-froze rather than
      the season's real decline.
    - "last": the latest downward crossing - the one immediately after
      the *last* time the series was at/above this threshold, i.e. the
      crossing tied to the final, sustained decline through that level.
      This is what tab 3's "melt start"/"melt-out" dates use when the UI's
      "last decline" option is selected, precisely to skip over an
      earlier re-snow bounce and report the decline that actually stuck.
    Both modes agree whenever a threshold is only crossed once (the
    common case for a genuinely monotonic melt).

    Returns a DataFrame, one row per threshold, with columns: threshold,
    crossing_date (NaT if not applicable), status ("crossed" / "not_reached"
    - snow_pct never got as high as this threshold at all, so there was
    nothing to cross down through / "still_covered" - reached but never
    dropped back below it by the last observation in the input, e.g. melt
    was still in progress when the fetched data ends / "insufficient_data"
    - fewer than two usable observations), and before_date/before_pct,
    after_date/after_pct (the two bracketing observations for a "crossed"
    row; NaT/NaN for every other status). See MELT_STATUS_LABELS for
    display-ready text."""
    if thresholds is None:
        thresholds = DEFAULT_MELT_THRESHOLDS
    if mode not in ("first", "last"):
        raise ValueError(f"snow_threshold_crossings: mode must be 'first' or 'last', got {mode!r}")
    d = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates).values), "pct": pd.Series(snow_pct).values})
    d = d.dropna(subset=["date", "pct"]).sort_values("date").reset_index(drop=True)

    rows = []
    if len(d) < 2:
        for t in thresholds:
            rows.append(dict(
                threshold=t, crossing_date=pd.NaT, status="insufficient_data",
                before_date=pd.NaT, before_pct=np.nan, after_date=pd.NaT, after_pct=np.nan,
            ))
        return pd.DataFrame(rows)

    for t in thresholds:
        crossing_date = pd.NaT
        before_date = after_date = pd.NaT
        before_pct = after_pct = np.nan
        status = "not_reached" if d["pct"].max() < t else "still_covered"
        if status == "still_covered":
            candidates = [
                i for i in range(len(d) - 1)
                if float(d["pct"].iloc[i]) >= t > float(d["pct"].iloc[i + 1])
            ]
            if candidates:
                i = candidates[0] if mode == "first" else candidates[-1]
                p0, p1 = float(d["pct"].iloc[i]), float(d["pct"].iloc[i + 1])
                d0, d1 = d["date"].iloc[i], d["date"].iloc[i + 1]
                frac = (p0 - t) / (p0 - p1) if p0 != p1 else 0.0
                crossing_date = d0 + (d1 - d0) * frac
                before_date, before_pct = d0, p0
                after_date, after_pct = d1, p1
                status = "crossed"
        rows.append(dict(
            threshold=t, crossing_date=crossing_date, status=status,
            before_date=before_date, before_pct=before_pct, after_date=after_date, after_pct=after_pct,
        ))
    return pd.DataFrame(rows)


def persistent_valid_fraction(
    scene_cache: dict, product_code: str, metric: str = "NDSI", polygon_mask: Optional[np.ndarray] = None,
):
    """Diagnostic for tab 3's "why is peak_snow_pct capped at the same
    value every year" investigation. Stacks every cached scene's `metric`
    array (NaN outside that metric's own valid-pixel mask - see
    SceneCacheEntry.metrics) for one product, and returns, per pixel, the
    fraction of those scenes where the pixel was valid: 0 means never
    valid across the *entire* fetched time range regardless of weather -
    a structurally excluded pixel (e.g. permanent terrain shadow fmask
    keeps misclassifying as cloud shadow, or a strip of the polygon
    outside the product's usable data footprint); 1 means always valid.
    A pixel that's merely cloud-affected sometimes will sit somewhere in
    between, not at 0.

    `polygon_mask` (optional) - a boolean 2D array, True = inside the
    polygon, on the same grid as the returned array (pass
    AnalysisResult.polygon_masks.get(product_code)) - when given, every
    pixel outside it is set to NaN in the returned array. Without this,
    the buffered viewing area around a polygon (cropped for landscape
    context, not clipped to the polygon's actual shape - see
    SCENE_VIEW_BUFFER_M) reads as "never valid" too, since it's NaN in
    every scene simply for being outside the polygon - which has nothing
    to do with cloud/shadow/footprint quality and can visually swamp a
    real, much smaller structural issue inside the polygon itself,
    especially for a narrow/elongated polygon where the background vastly
    outnumbers the polygon's own pixels.

    Returns (freq_array, transform, crs, n_scenes_used) - freq_array is a
    2D float32 array on that product's grid - or (None, None, None, 0) if
    no cached scene has this metric/product (e.g. nothing was ever fetched
    for it, or every scene's array had a mismatched shape)."""
    arrays = []
    transform = crs = None
    ref_shape = None
    for entry in scene_cache.values():
        if entry.product_code != product_code:
            continue
        arr = entry.metrics.get(metric) if entry.metrics else None
        if arr is None:
            continue
        if ref_shape is None:
            ref_shape = arr.shape
            transform, crs = entry.transform, entry.crs
        if arr.shape != ref_shape:
            continue
        arrays.append(np.isfinite(arr))
    if not arrays:
        return None, None, None, 0
    freq = np.stack(arrays).mean(axis=0).astype("float32")
    if polygon_mask is not None:
        pm_for_freq = polygon_mask
        # `polygon_mask` is always full-resolution (AnalysisResult.polygon_masks),
        # but `freq` may be block-mean downsampled (see
        # DEFAULT_RASTER_DOWNSAMPLE_FACTOR) - downsample the mask the same
        # way (via "any pixel in the block was inside the polygon", the
        # boolean analogue of block-mean) so the two still line up, rather
        # than silently skipping the mask whenever downsampling is on.
        if pm_for_freq.shape != freq.shape and freq.shape[0] > 0 and freq.shape[1] > 0:
            factor_h = round(pm_for_freq.shape[0] / freq.shape[0])
            factor_w = round(pm_for_freq.shape[1] / freq.shape[1])
            if factor_h > 1 and factor_h == factor_w:
                pm_for_freq = _downsample_mean(pm_for_freq.astype("float32"), factor_h) > 0
        if pm_for_freq.shape == freq.shape:
            freq = np.where(pm_for_freq, freq, np.nan)
    return freq, transform, crs, len(arrays)


def filter_scene_df(
    scene_df: pd.DataFrame,
    min_valid_pct: float = 0.0,
    max_snow_pct: float = 100.0,
    max_cloud_pct: float = 100.0,
    max_smoke_pct: float = 100.0,
    excluded_keys: Optional[set] = None,
    metric: Optional[str] = None,
    snow_method: str = DEFAULT_SNOW_METHOD,
) -> pd.DataFrame:
    """Applies the same quality gate `recompute_pixel_rasters()` uses to
    build the pixel raster, so the time-series/phenology plots and the
    raster stay in sync. Rows that never had any valid pixels at all for
    the metric being viewed are always dropped here - there's nothing to
    plot for them regardless of thresholds.

    `metric="NDSI"` switches the valid-pixel gate to NDSI's own, broader
    columns (n_valid_px_ndsi/pct_valid_ndsi) instead of the vegetation-index
    ones (n_valid_px/pct_valid) - NDSI's mask doesn't exclude fmask's snow
    class, so a heavily snow-covered scene the vegetation gate hard-drops
    can still pass here and show up when NDSI is the selected metric. Any
    other (or no) metric uses the original vegetation-index gate.
    `snow_method` picks which snow_pct estimate `max_snow_pct` is checked
    against - see `_snow_pct_for_method()`."""
    excluded_keys = excluded_keys or set()
    if len(scene_df) == 0:
        return scene_df
    keys = scene_df["product"] + "::" + scene_df["item_id"].astype(str)
    use_ndsi_gate = metric == "NDSI" and "n_valid_px_ndsi" in scene_df.columns
    valid_px_col = "n_valid_px_ndsi" if use_ndsi_gate else "n_valid_px"
    pct_valid_col = (
        "pct_valid_ndsi" if use_ndsi_gate and "pct_valid_ndsi" in scene_df.columns else "pct_valid"
    )
    snow_series = _snow_pct_for_method(scene_df, snow_method)
    keep = (
        (scene_df[valid_px_col] > 0)
        & (scene_df[pct_valid_col] >= min_valid_pct)
        & (snow_series <= max_snow_pct)
        & (scene_df.get("cloud_pct", 0.0) <= max_cloud_pct)
        & (scene_df.get("smoke_pct", 0.0) <= max_smoke_pct)
        & (~keys.isin(excluded_keys))
    )
    return scene_df[keep].copy()


def filter_uploaded_df(
    df: pd.DataFrame,
    min_valid_pct: float = 0.0,
    max_snow_pct: float = 100.0,
    max_cloud_pct: float = 100.0,
    max_smoke_pct: float = 100.0,
    months: Optional[list] = None,
    metric: Optional[str] = None,
    snow_method: str = DEFAULT_SNOW_METHOD,
) -> pd.DataFrame:
    """Same quality gate as `filter_scene_df()`, for tab 4's uploaded CSVs
    rather than a live `AnalysisResult`. Deliberately more forgiving:
    per the README, tab 4 accepts any per-scene CSV "as long as the columns
    match" - which could be an older export missing a column this app added
    later (e.g. smoke_pct, or the NDSI columns added after it), or a
    hand-edited file without n_valid_px/product/item_id at all - so every
    check here is skipped (not enforced, not an error) if its column isn't
    present, rather than requiring the full schema `filter_scene_df()` can
    assume from a fresh run. Expects a `month` column (1-12) already
    computed from `datetime` if filtering by month; `months=None` means no
    month filtering. `metric="NDSI"` uses n_valid_px_ndsi/pct_valid_ndsi
    instead of n_valid_px/pct_valid where present, same as
    `filter_scene_df()`."""
    if len(df) == 0:
        return df
    keep = pd.Series(True, index=df.index)
    use_ndsi_gate = metric == "NDSI" and "n_valid_px_ndsi" in df.columns
    valid_px_col = "n_valid_px_ndsi" if use_ndsi_gate else "n_valid_px"
    pct_valid_col = "pct_valid_ndsi" if use_ndsi_gate and "pct_valid_ndsi" in df.columns else "pct_valid"
    if valid_px_col in df.columns:
        keep &= df[valid_px_col] > 0
    if pct_valid_col in df.columns:
        keep &= df[pct_valid_col] >= min_valid_pct
    if "snow_pct" in df.columns or "snow_pct_fmask" in df.columns:
        keep &= _snow_pct_for_method(df, snow_method) <= max_snow_pct
    if "cloud_pct" in df.columns:
        keep &= df["cloud_pct"] <= max_cloud_pct
    if "smoke_pct" in df.columns:
        keep &= df["smoke_pct"] <= max_smoke_pct
    if months is not None and "month" in df.columns:
        keep &= df["month"].isin(months)
    return df[keep].copy()


def recompute_pixel_rasters(
    result: AnalysisResult,
    min_valid_pct: float = 0.0,
    max_snow_pct: float = 100.0,
    max_cloud_pct: float = 100.0,
    max_smoke_pct: float = 100.0,
    excluded_keys: Optional[set] = None,
    snow_method: str = DEFAULT_SNOW_METHOD,
) -> dict:
    """Rebuilds the per-pixel mean raster for every (product, metric) from
    `result.scene_cache`, restricted to scenes that pass the given quality
    thresholds and aren't in `excluded_keys` - purely in-memory, no DEA
    request, so this is cheap enough to call on every Streamlit rerun as
    the user adjusts filters or excludes/re-includes scenes.

    NDSI's raster is gated separately from the other four metrics, using
    filter_scene_df(..., metric="NDSI")'s broader kept-scene set - a scene
    that fails the vegetation-index valid-pixel gate (e.g. fully
    snow-covered) can still be a perfectly good NDSI scene, so it shouldn't
    be silently missing from the NDSI raster just because it's excluded
    from the NDVI/tasselled-cap ones."""
    excluded_keys = excluded_keys or set()
    veg_kept_df = filter_scene_df(
        result.scene_df, min_valid_pct, max_snow_pct, max_cloud_pct, max_smoke_pct,
        excluded_keys, metric=None, snow_method=snow_method,
    )
    veg_kept_keys = set(scene_key(r.product, r.item_id) for r in veg_kept_df.itertuples())
    ndsi_kept_df = filter_scene_df(
        result.scene_df, min_valid_pct, max_snow_pct, max_cloud_pct, max_smoke_pct,
        excluded_keys, metric="NDSI", snow_method=snow_method,
    )
    ndsi_kept_keys = set(scene_key(r.product, r.item_id) for r in ndsi_kept_df.itertuples())

    accum: dict = {}
    for key, entry in result.scene_cache.items():
        for metric, arr in entry.metrics.items():
            kept_keys = ndsi_kept_keys if metric == "NDSI" else veg_kept_keys
            if key not in kept_keys:
                continue
            acc_key = (entry.product_code, metric)
            acc = accum.get(acc_key)
            if acc is None:
                acc = dict(
                    sum=np.zeros(arr.shape, dtype="float64"),
                    count=np.zeros(arr.shape, dtype="int32"),
                    transform=entry.transform,
                    crs=entry.crs,
                )
                accum[acc_key] = acc
            finite = np.isfinite(arr)
            acc["sum"][finite] += arr[finite]
            acc["count"][finite] += 1

    pixel_rasters = {}
    for acc_key, acc in accum.items():
        with np.errstate(invalid="ignore"):
            mean = np.where(acc["count"] > 0, acc["sum"] / np.maximum(acc["count"], 1), np.nan)
        pixel_rasters[acc_key] = dict(
            mean=mean.astype("float32"), n_scenes=acc["count"],
            transform=acc["transform"], crs=acc["crs"],
        )
    return pixel_rasters


def fetch_single_scene(
    result: AnalysisResult, product_code: str, item_id: str, buffer_m: Optional[float] = None
) -> dict:
    """Re-fetches and clips just one already-known scene (looked up from
    `result.stac_items`, populated by run_analysis) - used by tab 3's
    "view this scene" feature. Returns dict(rgb, metrics, valid_mask,
    polygon_mask, transform, crs); `rgb` is the raw (unscaled) red/green/
    blue stack (h, w, 3) - deliberately NOT masked to only-valid pixels,
    so cloud/haze/smoke is visible in the true-colour view rather than
    hidden by the same mask that excludes it from the index statistics.
    `metrics` includes "NDSI" (computed over its own broader mask - see
    NDSI_VALID_FMASK_CLASSES) alongside the vegetation indices, whenever
    green/swir1 are available, regardless of whether the vegetation-index
    mask found any valid pixels for this scene.

    Unlike the rest of this module, which only ever needs pixels strictly
    inside the polygon, the returned raster is clipped to a square
    `buffer_m` metres wider than the polygon on every side (see
    `_buffered_square_gdf`), not the polygon's own tight bounding box -
    this is specifically for visual landscape context around the polygon
    when a user inspects one scene, not for any statistic. The polygon
    mask (`polygon_mask`, used to draw/exclude the "in polygon" area) is
    still computed from the polygon's own true geometry, unaffected by the
    wider fetch extent. `buffer_m=None` (the default) aims for a 2km x 2km
    square via `_scene_view_buffer_m()`, falling back to a flat 200m buffer
    once the polygon itself is already that big; pass an explicit value
    (0 for the old tight-crop behaviour, or any other number) to override."""
    with open(result.products_json_path) as f:
        all_products = json.load(f)
    params = all_products[product_code]
    canonical_bands = params["canonical_bands"]
    nodata_val = params["nodata"]

    item = result.stac_items.get(product_code, {}).get(item_id)
    if item is None:
        raise ValueError(f"Scene '{item_id}' not found for {product_code} in this analysis run.")

    roi_gdf = gpd.GeoDataFrame(geometry=[result.polygon_geom_wgs84], crs=WGS84)
    roi_projected = roi_gdf.to_crs(result.target_crs)
    if buffer_m is None:
        buffer_m = _scene_view_buffer_m(roi_gdf, result.target_crs)
    fetch_gdf = _buffered_square_gdf(roi_gdf, result.target_crs, buffer_m) if buffer_m > 0 else roi_gdf
    region_mgr = _SinglePolygonRegionManager(fetch_gdf, region_name=result.polygon_uid)
    product_mgr = RSDataProductManager(product_code, product_param_path=result.products_json_path)
    work_dir = tempfile.mkdtemp(prefix="peatland_trends_scene_")
    stac_mgr = STACDataManager(root_dir=work_dir)

    xr_ds = stac_mgr.stac_items_to_xrdataset([item], product_mgr, region_mgr, target_crs=result.target_crs)
    processed = product_mgr.make_product_dask(xr_ds, region_mgr, compute=True)
    if "time" not in processed.dims:
        processed = processed.expand_dims("time")
    scene = processed.isel(time=0)

    ref_band = canonical_bands["red"]
    # Computed from the polygon's own true geometry (not the buffered fetch
    # extent) - kept for anyone who wants it later, but deliberately NOT
    # used below to restrict rgb/metrics to just the polygon. The point of
    # the buffered fetch is landscape context, so both views cover the
    # whole buffered square; the polygon itself is drawn as a vector
    # outline over the top by the caller (app.py), not carved out of the
    # raster here.
    pm = _polygon_pixel_mask(roi_projected, scene[ref_band])

    raw_bands = {}
    for canon, varname in canonical_bands.items():
        if canon == "fmask" or varname not in scene.data_vars:
            continue
        raw_bands[canon] = np.asarray(scene[varname].values, dtype="float64")

    # `raster_valid`: not nodata / finite, across the whole buffered extent
    # - used for the true-colour view, which deliberately still shows
    # cloud/haze/smoke rather than masking it out (see this function's
    # docstring). `metric_valid` additionally excludes cloud/shadow/snow
    # per fmask, since an index computed over those pixels isn't meaningful
    # - used for the metric-layer view.
    #
    # The per-band `arr != nodata_val` check alone isn't reliable for the
    # buffered surround this function fetches: it only catches no-data
    # pixels whose DN happens to equal this product's *declared* nodata
    # value, but a scene's actual no-coverage fill (e.g. the buffer square
    # extending past that scene's real swath edge) doesn't always use that
    # exact sentinel. Left unfiltered, those pixels leaked into
    # raster_valid as "valid" and then into the true-colour percentile
    # stretch (raster_viz.rgb_to_png_data_uri), washing the whole image out
    # towards white. fmask's own no-data class (0) doesn't have that
    # ambiguity, so it's used here as a second, more reliable check -
    # without touching classes 2/3/4 (cloud/shadow/snow), which should stay
    # visible in the true-colour view.
    raster_valid = np.ones(scene[ref_band].shape, dtype=bool)
    for arr in raw_bands.values():
        raster_valid = raster_valid & (arr != nodata_val) & np.isfinite(arr)
    fmask_var = canonical_bands.get("fmask")
    fmask_arr = None
    if fmask_var and fmask_var in scene.data_vars:
        fmask_arr = np.asarray(scene[fmask_var].values)
        raster_valid = raster_valid & (fmask_arr != NODATA_FMASK_CLASS)
    metric_valid = raster_valid
    metric_valid_ndsi = raster_valid
    if fmask_arr is not None:
        metric_valid = raster_valid & np.isin(fmask_arr, FMASK_VALID_CLASSES)
        # Broader than metric_valid - keeps the snow class, so NDSI can
        # still be shown for a scene the vegetation-index mask excludes
        # entirely (e.g. fully snow-covered). See NDSI_VALID_FMASK_CLASSES.
        metric_valid_ndsi = raster_valid & np.isin(fmask_arr, NDSI_VALID_FMASK_CLASSES)

    metrics = {}
    if metric_valid.sum() > 0:
        refl = {c: np.where(metric_valid, a / REFLECTANCE_SCALE, np.nan) for c, a in raw_bands.items()}
        metrics = compute_indices(refl)
    if metric_valid_ndsi.sum() > 0 and "green" in raw_bands and "swir1" in raw_bands:
        refl_ndsi = {
            "green": np.where(metric_valid_ndsi, raw_bands["green"] / REFLECTANCE_SCALE, np.nan),
            "swir1": np.where(metric_valid_ndsi, raw_bands["swir1"] / REFLECTANCE_SCALE, np.nan),
        }
        metrics["NDSI"] = compute_ndsi(refl_ndsi)

    rgb = None
    if all(b in raw_bands for b in ("red", "green", "blue")):
        rgb = np.stack([raw_bands["red"], raw_bands["green"], raw_bands["blue"]], axis=-1)
        rgb = np.where(raster_valid[..., None], rgb, np.nan)

    return dict(
        rgb=rgb, metrics=metrics, valid_mask=metric_valid, polygon_mask=pm,
        transform=scene.rio.transform(), crs=scene.rio.crs, buffer_m=buffer_m,
    )
