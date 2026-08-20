"""
Turns a pixel-mean raster (a 2D numpy array + affine transform + CRS, as
produced by `analysis.run_analysis`'s `pixel_rasters`) into a
georeferenced, colour-mapped PNG that can be dropped onto a folium map as
an `ImageOverlay` on top of satellite imagery, plus a matching legend.

folium's `ImageOverlay` expects a simple axis-aligned lat/lon bounding box
(no rotation), so the raster - which is on a projected UTM grid - is
reprojected to EPSG:4326 first via rasterio.warp. For the polygon sizes
this app deals with (a few hectares up to a few hundred, at most) that
reprojection introduces no meaningful distortion.
"""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject


def _get_cmap(name):
    """matplotlib >= 3.9 removed matplotlib.cm.get_cmap(); matplotlib.colormaps[name]
    is the replacement (available since matplotlib 3.5). Falls back to the old
    API for older installs where `matplotlib.colormaps` doesn't exist."""
    try:
        return mpl.colormaps[name]
    except AttributeError:  # pragma: no cover - matplotlib < 3.5
        import matplotlib.cm as cm

        return cm.get_cmap(name)


def reproject_to_wgs84(arr: np.ndarray, transform, crs):
    """Reprojects a single-band float array on a projected grid to
    EPSG:4326. Returns (reprojected_array, bounds) where bounds is
    [[south, west], [north, east]] - the format folium.ImageOverlay wants."""
    src_crs = crs if isinstance(crs, CRS) else CRS.from_user_input(crs)
    dst_crs = CRS.from_epsg(4326)
    height, width = arr.shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, left=left, bottom=bottom, right=right, top=top
    )
    dst_arr = np.full((dst_height, dst_width), np.nan, dtype="float32")
    reproject(
        source=arr.astype("float32"),
        destination=dst_arr,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    dst_left, dst_top = dst_transform * (0, 0)
    dst_right, dst_bottom = dst_transform * (dst_width, dst_height)
    bounds = [[dst_bottom, dst_left], [dst_top, dst_right]]
    return dst_arr, bounds


def array_to_png_data_uri(arr: np.ndarray, cmap_name: str, vmin: float, vmax: float, opacity: float = 0.85) -> str:
    """Colour-maps a float array (NaN = transparent) to a base64 PNG data
    URI suitable for folium.raster_layers.ImageOverlay's `image` arg."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = (vmin - 0.5 if np.isfinite(vmin) else 0.0), (vmax + 0.5 if np.isfinite(vmax) else 1.0)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = _get_cmap(cmap_name)
    rgba = cmap(norm(arr))
    rgba[..., 3] = np.where(np.isnan(arr), 0.0, opacity)
    rgba = np.nan_to_num(rgba, nan=0.0)
    rgba_uint8 = (rgba * 255).astype("uint8")
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def colormap_legend(cmap_name: str, vmin: float, vmax: float, caption: str):
    """Returns a branca.colormap.LinearColormap matching `cmap_name`, ready
    to `.add_to(folium_map)` as a legend."""
    import branca.colormap as bcm

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = (vmin - 0.5 if np.isfinite(vmin) else 0.0), (vmax + 0.5 if np.isfinite(vmax) else 1.0)
    cmap = _get_cmap(cmap_name)
    colors = [mcolors.to_hex(cmap(x)) for x in np.linspace(0, 1, 9)]
    legend = bcm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=caption)
    return legend


def raster_overlay_layer(
    mean_arr: np.ndarray, transform, crs, metric_label: str, colorscale: str = "RdYlGn",
    vmin: float | None = None, vmax: float | None = None, opacity: float = 0.85,
):
    """Builds everything needed to show `mean_arr` on a folium map: a
    reprojected+colour-mapped PNG data URI, its lat/lon bounds, and a
    matching legend. Returns (data_uri, bounds, legend, vmin, vmax) or
    None if the array has no finite values at all.

    By default the colour scale autoscales to this array's own min/max,
    which is fine for the tasselled-cap indices (no fixed natural range)
    but misleading for NDVI: a healthy-vegetation polygon might only ever
    show values in a tight 0.55-0.7 band, and autoscaling that tiny window
    across the full colour ramp makes normal seasonal variation look like
    a dramatic swing. Pass explicit `vmin`/`vmax` (the app passes -1/1 for
    NDVI, its canonical range) to pin the scale instead.

    `opacity` (0-1) is exposed so the caller can let a user fade the raster
    to see the satellite basemap underneath - useful in tab 3's single-scene
    viewer, where several scenes' rasters can be layered over the same spot."""
    if not np.any(np.isfinite(mean_arr)):
        return None
    if vmin is None:
        vmin = float(np.nanmin(mean_arr))
    if vmax is None:
        vmax = float(np.nanmax(mean_arr))
    dst_arr, bounds = reproject_to_wgs84(mean_arr, transform, crs)
    data_uri = array_to_png_data_uri(dst_arr, colorscale, vmin, vmax, opacity=opacity)
    legend = colormap_legend(colorscale, vmin, vmax, metric_label)
    return data_uri, bounds, legend, vmin, vmax


def reproject_multiband_to_wgs84(arr_hwc: np.ndarray, transform, crs):
    """Same as reproject_to_wgs84 but for a (h, w, bands) array - each band
    is reprojected onto the same destination grid and re-stacked. Returns
    (reprojected_hwc_array, bounds)."""
    n_bands = arr_hwc.shape[-1]
    dst_bands = []
    bounds = None
    for b in range(n_bands):
        dst_band, bounds = reproject_to_wgs84(arr_hwc[..., b], transform, crs)
        dst_bands.append(dst_band)
    return np.stack(dst_bands, axis=-1), bounds


def rgb_to_png_data_uri(
    rgb_raw: np.ndarray, opacity: float = 1.0, low_pct: float = 2.0, high_pct: float = 98.0,
    gamma: float = 1.8,
) -> str:
    """Percentile-stretches a raw (h, w, 3) red/green/blue DN array (NaN =
    outside the polygon / no data) per band into an 8-bit true-colour PNG
    data URI. Each band is stretched independently using only its own
    finite values, so this works whether `rgb_raw` has already been
    reprojected (still float, still NaN outside the polygon) or not.

    `gamma` is applied after the linear percentile stretch, as
    `out ** (1/gamma)` - this brightens shadows/midtones without moving the
    black/white points (0 stays 0, 1 stays 1 either way). It matters
    because the 2nd-98th percentile clip is set by whatever's brightest/
    darkest in the *whole* buffered view, not just the polygon: a scene
    with a modest amount of snow (bright) surrounded by a much larger area
    of dense/shadowed vegetation (dark) pulls the upper clip point up,
    which linearly compresses that larger dark area down towards black
    with very little visible tonal separation. gamma=1.0 leaves the plain
    linear stretch as-is; higher values (this defaults to 1.8) lift that
    dark majority back into a visible range without touching how bright
    snow/highlights already look."""
    h, w, _ = rgb_raw.shape
    out = np.zeros((h, w, 3), dtype="float32")
    for b in range(3):
        band = rgb_raw[..., b]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1.0
        stretched = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
        if gamma and gamma != 1.0:
            stretched = np.power(stretched, 1.0 / gamma)
        out[..., b] = stretched
    alpha = np.where(np.all(np.isfinite(rgb_raw), axis=-1), opacity, 0.0)
    rgba = np.concatenate([out, alpha[..., None]], axis=-1)
    rgba = np.nan_to_num(rgba, nan=0.0)
    rgba_uint8 = (rgba * 255).astype("uint8")
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def rgb_overlay_layer(
    rgb_raw: np.ndarray, transform, crs, low_pct: float = 2.0, high_pct: float = 98.0, opacity: float = 1.0,
    gamma: float = 1.8,
):
    """Builds a true-colour PNG overlay + bounds for a single scene's raw
    (h, w, 3) red/green/blue array, for display on a folium satellite map
    - see `analysis.fetch_single_scene()`. Returns (data_uri, bounds) or
    None if there's nothing finite to show at all.

    `opacity` (0-1) - see raster_overlay_layer's docstring for why this is
    exposed here too. `gamma` - see `rgb_to_png_data_uri()`'s docstring."""
    if not np.any(np.isfinite(rgb_raw)):
        return None
    dst_rgb, bounds = reproject_multiband_to_wgs84(rgb_raw, transform, crs)
    data_uri = rgb_to_png_data_uri(dst_rgb, opacity=opacity, low_pct=low_pct, high_pct=high_pct, gamma=gamma)
    return data_uri, bounds
