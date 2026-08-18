"""
Core analysis pipeline: for one selected polygon, fetches Landsat and/or
Sentinel-2 surface reflectance scenes from Digital Earth Australia's STAC
catalogue and computes per-pixel NDVI / Tasselled Cap Brightness /
Greenness / Wetness for each scene.

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
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime as _dt
from typing import Callable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from shapely.geometry import box

_MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_MODULES_DIR)
_VENDOR_DIR = os.path.join(_APP_DIR, "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from rs_data import STACDataManager  # noqa: E402
from rs_processing import RSDataProductManager  # noqa: E402

from .indices import compute_indices, METRICS  # noqa: E402
from .smoke_detection import (  # noqa: E402
    DEFAULT_SMOKE_THRESHOLD_DN,
    build_clear_sky_reference,
    scene_smoke_pct,
)

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


def _is_transient_stac_error(exc: Exception) -> bool:
    """True for errors worth retrying (server-side hiccups) rather than
    giving up on a chunk immediately."""
    msg = str(exc)
    return any(needle in msg for needle in ("502", "503", "504", "Timeout", "Timed out", "Connection"))


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
                if attempt < STAC_SEARCH_MAX_ATTEMPTS - 1 and _is_transient_stac_error(e):
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


# fetch_single_scene()'s default landscape-context buffer - see that
# function's docstring for why this is a square rather than just the
# polygon's own (often much smaller and more irregularly-shaped) bounding
# box padded out.
SCENE_VIEW_BUFFER_M = 200.0


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
    metrics: dict             # metric name -> 2D float32 array (NaN outside valid_mask) - kept, needed by recompute_pixel_rasters
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
    batch_size: int = 15,
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
    cached output without needing to re-fetch anything.

    The STAC *search* itself (finding which scenes exist, before fetching
    any of their pixels) is done a few years at a time and retried on
    transient server errors - see `_search_stac_items()` - since a long
    date range searched in one request is exactly the kind of thing DEA's
    STAC gateway occasionally 504s on, and losing an entire multi-decade
    search over one flaky request would be a bad trade for a retry that
    usually just works.
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

            try:
                xr_ds = stac_mgr.stac_items_to_xrdataset(
                    batch_items, product_mgr, region_mgr, target_crs=target_crs
                )
                processed = product_mgr.make_product_dask(xr_ds, region_mgr, compute=True)
            except Exception as e:
                warnings.append(f"{sensor_label} batch {b + 1}/{n_batches}: failed to fetch/process - {e}")
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

                valid = pm.copy()
                for arr in raw_bands.values():
                    valid = valid & (arr != nodata_val) & np.isfinite(arr)
                snow_px = 0
                cloud_px = 0
                fmask_arr = None
                if fmask_var and fmask_var in scene.data_vars:
                    fmask_arr = np.asarray(scene[fmask_var].values)
                    valid = valid & np.isin(fmask_arr, FMASK_VALID_CLASSES)
                    snow_px = int((pm & (fmask_arr == SNOW_FMASK_CLASS)).sum())
                    cloud_px = int((pm & np.isin(fmask_arr, CLOUD_FMASK_CLASSES)).sum())

                valid_px = int(valid.sum())
                pct_valid = (100.0 * valid_px / total_px) if total_px else 0.0
                snow_pct = (100.0 * snow_px / total_px) if total_px else 0.0
                cloud_pct = (100.0 * cloud_px / total_px) if total_px else 0.0

                metrics = {}
                if valid_px > 0:
                    refl = {
                        canon: np.where(valid, arr / REFLECTANCE_SCALE, np.nan)
                        for canon, arr in raw_bands.items()
                    }
                    metrics = compute_indices(refl)

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
                    n_polygon_px=total_px,
                    pct_valid=round(pct_valid, 1),
                    snow_pct=round(snow_pct, 1),
                    cloud_pct=round(cloud_pct, 1),
                    smoke_pct=0.0,  # filled in below, once every scene's been fetched
                )
                for m in METRICS:
                    arr = metrics.get(m)
                    if arr is not None and valid_px > 0:
                        row[f"{m}_mean"] = float(np.nanmean(arr))
                        row[f"{m}_std"] = float(np.nanstd(arr))
                    else:
                        row[f"{m}_mean"] = np.nan
                        row[f"{m}_std"] = np.nan
                scene_rows.append(row)

                if valid_px > 0:
                    scene_cache[scene_key(product_code, item_id)] = SceneCacheEntry(
                        product_code=product_code,
                        item_id=item_id,
                        valid_mask=valid,        # released below, once smoke_pct is filled in
                        polygon_mask=None,       # never used downstream - not worth keeping
                        metrics=metrics,
                        blue_dn=raw_bands.get("blue"),  # released below, once smoke_pct is filled in
                        transform=transform,
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
        products_json_path=products_json_path,
        fetch_params=dict(
            start_date=start_date, end_date=end_date, product_codes=list(product_codes),
            max_cloud=max_cloud, min_coverage_pct=min_coverage_pct,
            smoke_threshold_dn=smoke_threshold_dn, batch_size=batch_size,
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


def filter_scene_df(
    scene_df: pd.DataFrame,
    min_valid_pct: float = 0.0,
    max_snow_pct: float = 100.0,
    max_cloud_pct: float = 100.0,
    max_smoke_pct: float = 100.0,
    excluded_keys: Optional[set] = None,
) -> pd.DataFrame:
    """Applies the same quality gate `recompute_pixel_rasters()` uses to
    build the pixel raster, so the time-series/phenology plots and the
    raster stay in sync. Rows that never had any valid pixels at all
    (n_valid_px == 0) are always dropped here - there's nothing to plot for
    them regardless of thresholds."""
    excluded_keys = excluded_keys or set()
    if len(scene_df) == 0:
        return scene_df
    keys = scene_df["product"] + "::" + scene_df["item_id"].astype(str)
    keep = (
        (scene_df["n_valid_px"] > 0)
        & (scene_df["pct_valid"] >= min_valid_pct)
        & (scene_df["snow_pct"] <= max_snow_pct)
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
) -> pd.DataFrame:
    """Same quality gate as `filter_scene_df()`, for tab 4's uploaded CSVs
    rather than a live `AnalysisResult`. Deliberately more forgiving:
    per the README, tab 4 accepts any per-scene CSV "as long as the columns
    match" - which could be an older export missing a column this app added
    later (e.g. smoke_pct), or a hand-edited file without n_valid_px/
    product/item_id at all - so every check here is skipped (not enforced,
    not an error) if its column isn't present, rather than requiring the
    full schema `filter_scene_df()` can assume from a fresh run. Expects a
    `month` column (1-12) already computed from `datetime` if filtering by
    month; `months=None` means no month filtering."""
    if len(df) == 0:
        return df
    keep = pd.Series(True, index=df.index)
    if "n_valid_px" in df.columns:
        keep &= df["n_valid_px"] > 0
    if "pct_valid" in df.columns:
        keep &= df["pct_valid"] >= min_valid_pct
    if "snow_pct" in df.columns:
        keep &= df["snow_pct"] <= max_snow_pct
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
) -> dict:
    """Rebuilds the per-pixel mean raster for every (product, metric) from
    `result.scene_cache`, restricted to scenes that pass the given quality
    thresholds and aren't in `excluded_keys` - purely in-memory, no DEA
    request, so this is cheap enough to call on every Streamlit rerun as
    the user adjusts filters or excludes/re-includes scenes."""
    excluded_keys = excluded_keys or set()
    kept_df = filter_scene_df(
        result.scene_df, min_valid_pct, max_snow_pct, max_cloud_pct, max_smoke_pct, excluded_keys
    )
    kept_keys = set(scene_key(r.product, r.item_id) for r in kept_df.itertuples())

    accum: dict = {}
    for key, entry in result.scene_cache.items():
        if key not in kept_keys:
            continue
        for metric, arr in entry.metrics.items():
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
    result: AnalysisResult, product_code: str, item_id: str, buffer_m: float = SCENE_VIEW_BUFFER_M
) -> dict:
    """Re-fetches and clips just one already-known scene (looked up from
    `result.stac_items`, populated by run_analysis) - used by tab 3's
    "view this scene" feature. Returns dict(rgb, metrics, valid_mask,
    polygon_mask, transform, crs); `rgb` is the raw (unscaled) red/green/
    blue stack (h, w, 3) - deliberately NOT masked to only-valid pixels,
    so cloud/haze/smoke is visible in the true-colour view rather than
    hidden by the same mask that excludes it from the index statistics.

    Unlike the rest of this module, which only ever needs pixels strictly
    inside the polygon, the returned raster is clipped to a square
    `buffer_m` metres wider than the polygon on every side (see
    `_buffered_square_gdf`), not the polygon's own tight bounding box -
    this is specifically for visual landscape context around the polygon
    when a user inspects one scene, not for any statistic. The polygon
    mask (`polygon_mask`, used to draw/exclude the "in polygon" area) is
    still computed from the polygon's own true geometry, unaffected by the
    wider fetch extent. Pass `buffer_m=0` to fall back to the old
    tight-crop behaviour."""
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
    if fmask_arr is not None:
        metric_valid = raster_valid & np.isin(fmask_arr, FMASK_VALID_CLASSES)

    metrics = {}
    if metric_valid.sum() > 0:
        refl = {c: np.where(metric_valid, a / REFLECTANCE_SCALE, np.nan) for c, a in raw_bands.items()}
        metrics = compute_indices(refl)

    rgb = None
    if all(b in raw_bands for b in ("red", "green", "blue")):
        rgb = np.stack([raw_bands["red"], raw_bands["green"], raw_bands["blue"]], axis=-1)
        rgb = np.where(raster_valid[..., None], rgb, np.nan)

    return dict(
        rgb=rgb, metrics=metrics, valid_mask=metric_valid, polygon_mask=pm,
        transform=scene.rio.transform(), crs=scene.rio.crs,
    )
