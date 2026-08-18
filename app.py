"""
Peatland trend analysis - Streamlit app.

Pick a bog/fen polygon (from the two Victorian & NSW alpine peatland
layers bundled with this app) on a map, choose a date range and sensor(s),
and this fetches every matching Landsat/Sentinel-2 scene from Digital
Earth Australia and computes per-pixel NDVI / Tasselled Cap Brightness /
Greenness / Wetness for each scene.

Fetching (tab 2) and everything downstream of it (tab 3) are deliberately
split: tab 2 only controls what gets *fetched* from DEA (date range,
sensors, whole-tile cloud ceiling, tile-coverage requirement). Data-quality
filtering (per-polygon valid-pixel/snow/cloud/smoke thresholds, and
manually excluding individual scenes) happens live in tab 3, recomputed
instantly from cached per-scene arrays with no further DEA requests - see
`modules/analysis.py`'s module docstring for why this is safe to do for
polygons this small.

Tab 4 is a separate, self-contained CSV-comparison tool - it doesn't need
a live DEA connection or a selected polygon at all.

Run with: streamlit run app.py  (see run_app.bat / README.md for setup)
"""

from __future__ import annotations

import calendar
import colorsys
import json
import os
import re
import sys
import tempfile
from datetime import date

import folium
import geopandas as gpd
import jinja2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules import analysis, polygons, raster_viz, trends  # noqa: E402
from modules.indices import METRIC_LABELS, METRICS  # noqa: E402
from modules.smoke_detection import DEFAULT_SMOKE_THRESHOLD_DN  # noqa: E402

PRODUCTS_JSON = os.path.join(BASE_DIR, "config", "products.json")

LANDSAT5_LAUNCH = date(1984, 3, 1)  # Landsat 5 launched 1 March 1984
TODAY = date.today()

PRODUCT_CHOICES = [
    ("landsat_full", "Landsat (5/7/8/9, 30 m)"),
    ("sentinel_full", "Sentinel-2 (10 m)"),
]
PRODUCT_LABELS = dict(PRODUCT_CHOICES)

SOURCE_COLORS = {
    polygons.SOURCE_ALPINE_BOGS: "#7FB3D5",   # light blue
    polygons.SOURCE_SNOWY_MTNS: "#F5B041",    # amber
}
SELECTED_COLOR = "#E74C3C"  # red

LANDSAT_PIXEL_HA = (30 * 30) / 10000.0
SENTINEL_PIXEL_HA = (10 * 10) / 10000.0

# Tab 3's single-scene viewer: how many scenes' rasters can be held (and
# layered on the same map) at once. Each is small (a buffered square around
# a bog/fen-sized polygon, see analysis.SCENE_VIEW_BUFFER_M) so memory isn't
# the constraint - this is purely to keep the map/layer-control legible and
# the "which scene is which" bookkeeping simple for the user.
MAX_VIEWED_SCENES = 5

SENSOR_SYMBOLS = {"Landsat": "circle", "Sentinel-2": "diamond"}


def _month_color(m: int) -> str:
    h = (m - 1) / 12.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.62, 0.85)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


MONTH_COLORS = {m: _month_color(m) for m in range(1, 13)}

# The phenology plot runs July -> June ("water year") rather than
# Jan -> Dec, so a single alpine growing season (which spans the
# Dec/Jan calendar-year boundary in the Southern Hemisphere) appears as one
# unbroken block in the middle of the x-axis instead of being split across
# both ends of it - matters for both readability and for the trend line fit
# (a straight line across a Jan-Dec axis would otherwise connect December
# back to January the "wrong" way round).
WATER_MONTH_ORDER = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]


def to_water_month(calendar_month) -> int:
    """Calendar month (1-12, Jan=1) -> water-year position (1-12, Jul=1)."""
    return ((calendar_month - 7) % 12) + 1


def facet_rc(i: int, n_cols: int) -> tuple[int, int]:
    """0-based facet index -> (row, col) 1-based Plotly subplot coordinates,
    filling left-to-right then top-to-bottom - matches the order
    make_subplots(subplot_titles=...) lays facets out in, used by tab 4's
    per-dataset faceted trend/phenology plots."""
    return i // n_cols + 1, i % n_cols + 1


def build_categorical_color_map(values) -> dict:
    """Assigns a consistent colour to each unique value in `values` from
    Plotly's default qualitative palette. Used so e.g. 'Landsat' is the
    same colour in every facet of tab 4's comparison plots, rather than
    each facet's traces getting an arbitrary colour independently (the
    default when you add traces to different subplots one at a time via
    go.Figure/add_trace instead of letting a single px.scatter call assign
    them all at once)."""
    uniq = sorted({v for v in values if pd.notna(v)}, key=str)
    palette = px.colors.qualitative.Plotly
    return {v: palette[i % len(palette)] for i, v in enumerate(uniq)}


st.set_page_config(page_title="Peatland Trend Analysis", layout="wide")


# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading peatland polygon layers...")
def _load_polygons():
    gdf = polygons.load_polygons(BASE_DIR)
    disp = polygons.display_geometry(gdf)
    options = (gdf["uid"] + "  |  " + gdf["label"] + "  (" + gdf["source"] + ")").tolist()
    return gdf, disp, options


POLY_GDF, POLY_DISPLAY_GEOM, POLY_OPTIONS = _load_polygons()
UID_TO_OPTION = dict(zip(POLY_GDF["uid"], POLY_OPTIONS))
OPTION_TO_UID = {v: k for k, v in UID_TO_OPTION.items()}

_minx, _miny, _maxx, _maxy = POLY_GDF.total_bounds
ALL_POLYGONS_BOUNDS = [[_miny, _minx], [_maxy, _maxx]]  # [[south, west], [north, east]]


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def init_state():
    defaults = dict(
        selected_uid=None,
        show_alpine_bogs=True,
        show_snowy_mtns=True,
        map_v=0,
        result=None,
        result_uid=None,
        run_error=None,
        excluded_keys=set(),
        viewed_scenes=[],  # up to MAX_VIEWED_SCENES fetched scenes at a time - see tab 3's scene viewer
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def selected_row():
    if st.session_state.selected_uid is None:
        return None
    match = POLY_GDF.loc[POLY_GDF["uid"] == st.session_state.selected_uid]
    return match.iloc[0] if len(match) else None


def satellite_tile_layer():
    return folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satellite", overlay=False, control=False,
    )


class _MapVisibilityFix(folium.MacroElement):
    """Streamlit's st.tabs() keeps every tab's content mounted in the DOM
    all the time and just hides inactive panes with CSS - and switching
    tabs is a purely client-side action that never triggers a Streamlit
    script rerun. So a Folium map that happens to render (and call
    fit_bounds()) while its tab is hidden mounts into a 0-pixel-wide
    container, Leaflet computes its zoom from that, and it never gets a
    second chance once the tab becomes visible - the result is the
    "fully zoomed out, world map tiled sideways" view. A ResizeObserver on
    the map's own container reliably fires the moment the tab actually
    becomes visible (that's a real resize event, 0x0 -> real pixels), so
    this re-runs invalidateSize()+fitBounds() right then.

    This has to be a folium MacroElement added as a child of the map
    (`.add_to(m)`), NOT a plain `folium.Element` dropped into
    `m.get_root().html` (an earlier version of this fix did exactly that,
    and it silently never worked at all - however it flows, everything
    under `get_root().html` ends up passed to streamlit-folium's frontend
    as an `html` prop, which the frontend inserts via `.innerHTML =`,
    which never executes embedded `<script>` tags, in every browser).
    Elements added directly to the map, on the other hand, get their
    `script` macro pulled into the separate `script` string streamlit-
    folium *does* execute (via a dynamically created `<script>` element,
    which does run) - the same mechanism folium's own TileLayer/GeoJson
    etc. use internally, which is why using a proper MacroElement subclass
    here (rather than reaching for something not part of that mechanism)
    is what makes this actually work. Reference the map as
    `{{ this._parent.get_name() }}` in the template, same as any other
    folium element would - streamlit-folium substitutes its own fixed
    `map_div` variable name for whatever that resolves to, consistently
    across every script fragment including this one."""

    _template = jinja2.Template(
        """
        {% macro script(this, kwargs) %}
        (function() {
            function attachResizeFix() {
                var mapObj = {{ this._parent.get_name() }};
                var container = mapObj.getContainer();
                var ro = new ResizeObserver(function(entries) {
                    for (var i = 0; i < entries.length; i++) {
                        var rect = entries[i].contentRect;
                        if (rect.width > 0 && rect.height > 0) {
                            mapObj.invalidateSize();
                            mapObj.fitBounds({{ this.bounds_json }});
                        }
                    }
                });
                ro.observe(container);
            }
            attachResizeFix();
        })();
        {% endmacro %}
        """
    )

    def __init__(self, bounds):
        super().__init__()
        self._name = "MapVisibilityFix"
        self.bounds_json = json.dumps(bounds)


def fix_map_visibility_zoom(m: "folium.Map", bounds) -> None:
    """Call this after m.fit_bounds(bounds) for any map that lives inside a
    tab that might not be the active one when the map first mounts - see
    _MapVisibilityFix's docstring for why a plain folium.Element can't do
    this job. Harmless to call even for a map on the default-active tab."""
    _MapVisibilityFix(bounds).add_to(m)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Peatland Trend Analysis")

st.sidebar.markdown("**Map layers**")
st.session_state.show_alpine_bogs = st.sidebar.checkbox(
    f"{polygons.SOURCE_ALPINE_BOGS} ({(POLY_GDF['source'] == polygons.SOURCE_ALPINE_BOGS).sum()} polygons)",
    value=st.session_state.show_alpine_bogs,
)
st.session_state.show_snowy_mtns = st.sidebar.checkbox(
    f"{polygons.SOURCE_SNOWY_MTNS} ({(POLY_GDF['source'] == polygons.SOURCE_SNOWY_MTNS).sum()} polygons)",
    value=st.session_state.show_snowy_mtns,
)
st.sidebar.caption(
    "Both layers together are ~12,000 small polygons - the map can be slow to pan/zoom. "
    "Turning one layer off, or using the search box below, helps."
)

st.sidebar.markdown("**Find a polygon**")
current_option = UID_TO_OPTION.get(st.session_state.selected_uid)
picked_option = st.sidebar.selectbox(
    "Search by ID / name",
    options=[""] + POLY_OPTIONS,
    index=(POLY_OPTIONS.index(current_option) + 1) if current_option else 0,
    label_visibility="collapsed",
    placeholder="Type to search...",
)
if picked_option and OPTION_TO_UID.get(picked_option) != st.session_state.selected_uid:
    st.session_state.selected_uid = OPTION_TO_UID[picked_option]
    st.session_state.map_v += 1
    st.rerun()

sel = selected_row()
if sel is not None:
    st.sidebar.success(f"Selected: {sel['label']}")
    st.sidebar.caption(f"uid: {sel['uid']}  |  {sel['source']}  |  {sel['area_ha']:.3f} ha")
    if st.sidebar.button("Clear selection"):
        st.session_state.selected_uid = None
        st.session_state.result = None
        st.session_state.map_v += 1
        st.rerun()
else:
    st.sidebar.info("No polygon selected yet.")

with st.sidebar.expander("About this app", expanded=False):
    st.markdown(
        """
        Select a bog/fen polygon from the two peatland layers bundled with
        this app, then fetch and analyse Landsat/Sentinel-2 scenes from
        Digital Earth Australia over it. Computes NDVI, Tasselled Cap
        Brightness, Greenness and Wetness per scene, clipped to the
        polygon.

        Data access is anonymous/public DEA STAC - no personal account or
        API key needed. Scenes are fetched and processed in small batches
        so memory use stays bounded even over the full multi-decade
        archive.

        Built as a sibling of the `Satellite downloader` app in this same
        repo (shares its DEA access code, vendored into `vendor/`), driven
        by the analysis originally prototyped in this folder's R Shiny app
        (`global.R` / `ui.R` / `server.R`). Smoke/haze detection is adapted
        from `../Estuary openings local app`.
        """
    )


# --------------------------------------------------------------------------
# Map helpers (tab 1 - polygon picker)
# --------------------------------------------------------------------------

def build_map(bounds):
    """`bounds` is [[south, west], [north, east]]. Uses fit_bounds rather
    than a fixed center/zoom_start - more reliable with streamlit-folium,
    which can otherwise render the map fully zoomed out (world view) until
    the user interacts with it, since the map's pixel size isn't always
    known yet at the point zoom_start would be applied."""
    m = folium.Map(tiles=None, control_scale=True)
    satellite_tile_layer().add_to(m)
    m.fit_bounds(bounds)
    fix_map_visibility_zoom(m, bounds)  # this tab is no longer guaranteed to be the default-active one
    return m


def add_polygon_layer(m, gdf, geom, source_name, color, selected_uid):
    sub = gdf[gdf["source"] == source_name]
    if len(sub) == 0:
        return
    sub_geom = geom.loc[sub.index]
    gj = pd.concat([sub[["uid", "label", "source"]], sub_geom.rename("geometry")], axis=1)
    gj = gpd.GeoDataFrame(gj, geometry="geometry", crs=polygons.WGS84)

    def style_fn(feature):
        is_sel = feature["properties"]["uid"] == selected_uid
        return {
            "color": SELECTED_COLOR if is_sel else color,
            "weight": 3 if is_sel else 1,
            "fillColor": SELECTED_COLOR if is_sel else color,
            "fillOpacity": 0.55 if is_sel else 0.25,
        }

    folium.GeoJson(
        gj,
        name=source_name,
        style_function=style_fn,
        highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.6},
        tooltip=folium.GeoJsonTooltip(fields=["label"], labels=False, sticky=True),
        popup=folium.GeoJsonPopup(fields=["uid"], labels=False),
    ).add_to(m)


_UID_RE = re.compile(r"\b(AB|SM)-\d+\b")


def parse_uid_from_popup(text):
    if not text:
        return None
    match = _UID_RE.search(text)
    return match.group(0) if match else None


# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------

tab_home, tab_select, tab_run, tab_results, tab_compare = st.tabs(
    ["Home", "1. Select polygon", "2. Configure & run", "3. Results", "4. Trend comparisons"]
)

with tab_home:
    st.title("Peatland Trend Analysis")
    st.write(
        "This app tracks how vegetation and wetness in Victorian alpine bogs and Snowy Mountains "
        "fens have changed over time, using free satellite imagery from Digital Earth Australia "
        "(DEA) - Landsat back to March 1984, Sentinel-2 from 2015. Pick a bog or fen polygon, choose "
        "a date range and sensor(s), and it fetches every matching satellite scene, calculates "
        "vegetation (NDVI) and wetness/brightness/greenness (tasselled cap) indices for the polygon, "
        "and lets you explore long-term trends, seasonal patterns, and individual scenes."
    )
    st.caption(
        "DEA access is anonymous and public - no account, API key, or credential is needed anywhere "
        "in this app."
    )

    st.divider()
    st.subheader("How to use it")
    st.markdown(
        """
1. **Select polygon.** Click a polygon on the map (it highlights in red once selected), or use the
   searchable dropdown in the sidebar - with roughly 12,000 small polygons across both layers, the
   search box is often faster than clicking. Toggle the two layers on/off independently in the
   sidebar.
2. **Configure & run.** Choose a date range (bounded by Landsat 5's launch to today), tick Landsat,
   Sentinel-2, or both, and press **Run analysis**. This is the only step that talks to DEA -
   everything in tab 3 works instantly afterwards from what gets fetched here, with no further
   fetching. *Advanced fetch options* let you set a whole-tile cloud-cover ceiling, a minimum
   tile-coverage requirement, and a smoke/haze sensitivity - the defaults are sensible for most runs.
3. **Results.** Everything here updates live as you change a filter, exclude a scene, or switch
   metric - no re-fetch needed:
   - Data-quality filters (minimum valid pixels, maximum snow/cloud/smoke %) decide which fetched
     scenes count toward the raster and plots below.
   - A **pixel-mean raster** map shows the average value per pixel over your polygon, per sensor.
   - A **time series** and a **phenology plot** (seasonal pattern, by month) both support optional
     linear and GAM-smoothed trend lines.
   - Click a point on the time series to fetch and view that one scene's imagery (true colour or
     the current metric), download it as a GeoTIFF, or exclude it from the analysis.
   - A per-scene data table lets you review everything fetched and download it as CSV.
4. **Trend comparisons.** A separate tool that doesn't need a polygon selected at all - upload two
   or more per-scene CSVs (exported from tab 3) to compare trends across different polygons or runs,
   one facet per file, with the same filters and trend-line toggles as tab 3.
        """
    )

    st.divider()
    st.subheader("Good to know")
    st.markdown(
        """
- **"0% cloud cover" in tab 2 returning nothing?** That slider is DEA's whole-*tile* cloud cover
  (100km+ across), not your polygon - it's really just a fetch-volume limiter. Leave it generous
  (default 80%) and use tab 3's "maximum scene cloud cover over polygon" slider for real cloud
  filtering, which is computed from just your polygon.
- **Smoke/haze detection is an experimental heuristic**, not validated for every site - if tab 3
  flags a scene, check it with the single-scene viewer before trusting it, and adjust tab 2's
  sensitivity if needed.
- **The phenology plot runs July -> June**, not January -> December, so a Southern Hemisphere
  summer growing season shows as one unbroken block instead of being split across the year.
- **A dotted line at zero on tasselled cap wetness (TCW) plots** marks the rough threshold above
  which open water is typically present.
- **NDVI's raster legend is fixed to its canonical -1 to 1 range**, so colour is comparable across
  different runs and polygons rather than autoscaling to whatever narrow range your data happens to
  fall in. Other metrics still autoscale.
- **The scene viewer holds up to 5 scenes at once**, layered on the same map - use the layer control
  (top right of the map, once more than one scene is shown) or the "Remove" buttons above the map to
  toggle or drop individual scenes. Raster visibility and opacity are also adjustable, and each scene
  shows a 200 m buffer around the polygon (drawn as a dashed outline over the top, with an option to
  hide it) rather than just the polygon itself, for landscape context.
- **Long date ranges (e.g. 1985-present) are searched a few years at a time and retried** if DEA's
  STAC gateway times out, since that's more common on large multi-year searches. If a run still
  ends with a "STAC search failed" warning under tab 2, it's usually a transient gateway hiccup -
  just try again, or narrow the date range if it keeps happening.
        """
    )

with tab_select:
    st.write(
        "Click a polygon on the map to select it (or use the search box in the sidebar). "
        "The selected polygon is highlighted in red."
    )

    if sel is not None:
        minx, miny, maxx, maxy = sel.geometry.bounds
        # Small buffer so a tiny polygon doesn't fit_bounds to an
        # unreadably close-in view - at least ~150m of padding either side.
        pad_x = max((maxx - minx) * 0.6, 0.0015)
        pad_y = max((maxy - miny) * 0.6, 0.0015)
        view_bounds = [[miny - pad_y, minx - pad_x], [maxy + pad_y, maxx + pad_x]]
    else:
        view_bounds = ALL_POLYGONS_BOUNDS

    m = build_map(view_bounds)
    if st.session_state.show_alpine_bogs:
        add_polygon_layer(
            m, POLY_GDF, POLY_DISPLAY_GEOM, polygons.SOURCE_ALPINE_BOGS,
            SOURCE_COLORS[polygons.SOURCE_ALPINE_BOGS], st.session_state.selected_uid,
        )
    if st.session_state.show_snowy_mtns:
        add_polygon_layer(
            m, POLY_GDF, POLY_DISPLAY_GEOM, polygons.SOURCE_SNOWY_MTNS,
            SOURCE_COLORS[polygons.SOURCE_SNOWY_MTNS], st.session_state.selected_uid,
        )
    folium.LayerControl().add_to(m)

    map_data = st_folium(
        m,
        key=f"select_map_{st.session_state.map_v}",
        height=650,
        use_container_width=True,
        returned_objects=["last_object_clicked_popup"],
    )

    clicked_uid = parse_uid_from_popup((map_data or {}).get("last_object_clicked_popup"))
    if clicked_uid and clicked_uid != st.session_state.selected_uid and clicked_uid in OPTION_TO_UID.values():
        st.session_state.selected_uid = clicked_uid
        st.session_state.map_v += 1
        st.rerun()

# ==========================================================================
# Tab 2 - fetch from DEA (only params that require a re-fetch live here)
# ==========================================================================
with tab_run:
    if sel is None:
        st.warning("Select a polygon in tab 1 first.")
    else:
        st.subheader(f"Selected polygon: {sel['label']}")
        st.caption(f"uid: {sel['uid']}  |  {sel['source']}  |  area: {sel['area_ha']:.3f} ha")

        if sel["area_ha"] < LANDSAT_PIXEL_HA:
            st.info(
                f"This polygon ({sel['area_ha']:.3f} ha) is smaller than a single Landsat pixel "
                f"({LANDSAT_PIXEL_HA:.2f} ha) and "
                + ("smaller than a single Sentinel-2 pixel too " if sel["area_ha"] < SENTINEL_PIXEL_HA
                   else "close to a single Sentinel-2 pixel ")
                + f"({SENTINEL_PIXEL_HA:.2f} ha). Results are based on whichever pixels touch the "
                "polygon boundary (not just pixels centred inside it) but will reflect only 1-2 "
                "boundary-mixed pixels. Sentinel-2 is recommended over Landsat for polygons this small."
            )

        st.markdown("**Date range**")
        date_c1, date_c2 = st.columns(2)
        with date_c1:
            start_date = st.date_input(
                "Start date", value=LANDSAT5_LAUNCH, min_value=LANDSAT5_LAUNCH, max_value=TODAY
            )
        with date_c2:
            end_date = st.date_input(
                "End date", value=TODAY, min_value=LANDSAT5_LAUNCH, max_value=TODAY
            )

        st.markdown("**Satellite data**")
        prod_cols = st.columns(len(PRODUCT_CHOICES))
        chosen_products = []
        for col, (code, label) in zip(prod_cols, PRODUCT_CHOICES):
            with col:
                if st.checkbox(label, value=True, key=f"prod_{code}"):
                    chosen_products.append(code)
        if not chosen_products:
            st.caption("Select at least one of Landsat / Sentinel-2 above to enable Run.")

        with st.expander("Advanced fetch options", expanded=False):
            st.caption(
                "These only affect what gets *fetched* from DEA. Data-quality filtering (valid-pixel %, "
                "snow/cloud/smoke over the polygon itself, and excluding individual scenes) happens live "
                "in tab 3 after fetching, with no need to re-run this step."
            )
            max_cloud = st.slider(
                "Maximum whole-tile cloud cover (%)", min_value=0, max_value=100, value=80,
                help="DEA's own eo:cloud_cover field - the cloud percentage over the *entire* satellite "
                "tile (100km+ across for Landsat), not just your polygon. Setting this very low (e.g. 0%) "
                "often returns nothing, because almost every tile has cloud *somewhere* even when your "
                "polygon itself is perfectly clear. This is only a fetch-volume limiter; the real "
                "polygon-level cloud filter is in tab 3.",
            )
            min_coverage_pct = st.slider(
                "Minimum tile coverage of polygon (%)", min_value=0, max_value=100, value=100,
                help="Only fetch scenes whose tile fully (or mostly) overlaps the polygon - drops "
                "scenes where the polygon sits at the edge of a swath or straddles two tiles collected "
                "on different days, before any raster is fetched. Set to 0 to keep every scene "
                "regardless of coverage. 100% means 'fully covers' rather than mathematically exact "
                "containment, so it won't reject a genuinely fully-covering scene over floating-point "
                "noise in the overlap calculation.",
            )
            smoke_threshold_dn = st.number_input(
                "Smoke/haze detection sensitivity (blue-band DN)", min_value=50, max_value=2000,
                value=int(DEFAULT_SMOKE_THRESHOLD_DN), step=50,
                help="How far above a pixel's own historical clear-sky blue-band brightness (in raw DN, "
                "reflectance x10000) counts as a temporal anomaly - lower catches more (but risks false "
                "positives on naturally bright surfaces); higher catches only strong smoke/haze events. "
                "This is a starting point, not validated for these sites - use tab 3's 'view this scene' "
                "feature to sanity-check flagged scenes against their true-colour image.",
            )
            batch_size = st.number_input(
                "Scenes per batch", min_value=1, max_value=50, value=15,
                help="Scenes are fetched and processed this many at a time during the DEA request, so "
                "memory use stays bounded on long date ranges/busy polygons - useful if this app is "
                "running on a server with limited RAM.",
            )

        can_run = bool(chosen_products) and start_date <= end_date
        if start_date > end_date:
            st.error("Start date must be before end date.")

        if st.button("Run analysis", type="primary", disabled=not can_run):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_cb(msg, frac):
                status_text.write(msg)
                progress_bar.progress(frac)

            try:
                result = analysis.run_analysis(
                    polygon_geom_wgs84=sel.geometry,
                    polygon_uid=sel["uid"],
                    product_codes=chosen_products,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    products_json_path=PRODUCTS_JSON,
                    max_cloud=max_cloud,
                    min_coverage_pct=min_coverage_pct,
                    smoke_threshold_dn=float(smoke_threshold_dn),
                    batch_size=int(batch_size),
                    progress_cb=progress_cb,
                )
                st.session_state.result = result
                st.session_state.result_uid = sel["uid"]
                st.session_state.run_error = None
                st.session_state.excluded_keys = set()
                st.session_state.viewed_scenes = []
                status_text.write("Done.")
            except Exception as e:
                st.session_state.result = None
                st.session_state.run_error = str(e)
                st.exception(e)

        result = st.session_state.result
        if result is not None and st.session_state.result_uid == sel["uid"]:
            n_scenes = len(result.scene_df)
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Scenes fetched", n_scenes)
            if len(result.scene_df) > 0:
                sensors = result.scene_df["sensor"].value_counts().to_dict()
                m2.metric("Sensors", ", ".join(f"{k} ({v})" for k, v in sensors.items()))
                m3.metric(
                    "Date range covered",
                    f"{result.scene_df['date'].min()} → {result.scene_df['date'].max()}",
                )
            for code, area in result.polygon_area_ha.items():
                st.caption(f"{code}: polygon rasterised to {area:.3f} ha on this grid.")
            if len(result.scene_df) > 0:
                diag_bits = []
                for col, label in [("snow_pct", "snow"), ("cloud_pct", "cloud (over the polygon)"), ("smoke_pct", "smoke/haze")]:
                    if col in result.scene_df.columns:
                        n_flagged = int((result.scene_df[col] > 0).sum())
                        if n_flagged > 0:
                            diag_bits.append(f"{n_flagged} scene(s) with some {label} (max {result.scene_df[col].max():.0f}%)")
                if diag_bits:
                    st.caption("Diagnostics: " + "; ".join(diag_bits) + ". Adjust quality filters for these live in tab 3.")
            cache_mb = analysis.cache_size_bytes(result) / (1024 * 1024)
            st.caption(
                f"In-memory scene cache for this run: {cache_mb:.2f} MB "
                f"({n_scenes} scenes x this polygon's pixel footprint). This is what tab 3's live "
                "filters/exclusion work from without re-fetching - it's held for as long as this "
                "result stays in this browser session's memory, and is replaced (old data becomes "
                "garbage-collectable) as soon as you run a new polygon. On a shared server, total "
                "memory scales with concurrent sessions x each one's cache size, not with this run alone."
            )
            if result.warnings:
                with st.expander(f"{len(result.warnings)} warning(s) during this run", expanded=False):
                    for w in result.warnings:
                        st.warning(w)
            st.success("See tab 3 for results.")

# ==========================================================================
# Tab 3 - live-filterable results
# ==========================================================================
with tab_results:
    result = st.session_state.result
    if result is None or sel is None or st.session_state.result_uid != sel["uid"]:
        st.info("Run an analysis in tab 2 first.")
    elif len(result.scene_df) == 0:
        st.warning("No scenes matched the fetch filters for this polygon.")
    else:
        df_all = result.scene_df.copy()

        # ---- Live data-quality filters ---------------------------------
        st.subheader("Data quality filters")
        st.caption("These apply instantly to everything below, without re-fetching from DEA.")
        qc1, qc2 = st.columns(2)
        with qc1:
            min_valid_pct = st.slider("Minimum valid pixels per scene (%)", 0, 100, 50, key="qc_min_valid")
            max_snow_pct = st.slider(
                "Maximum scene snow cover (%)", 0, 100, 100, key="qc_max_snow",
                help="Snow pixels are always excluded from the index calculations themselves; this "
                "rejects whole scenes above a chosen snow extent.",
            )
        with qc2:
            max_cloud_pct = st.slider(
                "Maximum scene cloud cover over polygon (%)", 0, 100, 100, key="qc_max_cloud",
                help="Computed from fmask over just this polygon - unlike tab 2's whole-tile cloud "
                "filter, this is the one that actually reflects cloud over your bog.",
            )
            max_smoke_pct = st.slider(
                "Maximum scene smoke/haze cover (%)", 0, 100, 100, key="qc_max_smoke",
                help="Experimental temporal-anomaly detection (see tab 2's sensitivity setting) - "
                "catches some smoke/haze that fmask's cloud mask misses.",
            )

        excluded_keys = st.session_state.excluded_keys
        if excluded_keys:
            ec1, ec2 = st.columns([3, 1])
            ec1.caption(f"{len(excluded_keys)} scene(s) manually excluded (see 'Per-scene data' below).")
            if ec2.button("Clear exclusions"):
                st.session_state.excluded_keys = set()
                st.rerun()

        kept_df = analysis.filter_scene_df(
            df_all, min_valid_pct, max_snow_pct, max_cloud_pct, max_smoke_pct, excluded_keys
        )
        pixel_rasters = analysis.recompute_pixel_rasters(
            result, min_valid_pct, max_snow_pct, max_cloud_pct, max_smoke_pct, excluded_keys
        )
        st.caption(f"{len(kept_df)} of {len(df_all)} fetched scenes currently pass these filters.")

        st.divider()
        available_products = sorted({code for (code, _m) in pixel_rasters.keys()})
        metric = st.selectbox(
            "Metric", METRICS, format_func=lambda m: METRIC_LABELS.get(m, m), key="tab3_metric_select"
        )
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        colorscale = "RdYlGn" if metric in ("NDVI", "TCG") else "YlGnBu"

        # ---- Pixel-mean raster ------------------------------------------
        st.subheader("Pixel-mean raster")
        if not available_products:
            st.info("No scenes currently pass the quality filters, so no pixel raster could be built.")
        else:
            raster_product = st.radio(
                "Sensor (rasters are shown per-sensor since Landsat and Sentinel-2 grids differ)",
                available_products, format_func=lambda c: PRODUCT_LABELS.get(c, c), horizontal=True,
                key="raster_product",
            )
            raster = pixel_rasters.get((raster_product, metric))
            if raster is None:
                st.info(f"No {PRODUCT_LABELS.get(raster_product, raster_product)} raster available for this metric.")
            else:
                mean_arr = raster["mean"]
                n_scenes_arr = raster["n_scenes"]
                raster_vmin, raster_vmax = (-1.0, 1.0) if metric == "NDVI" else (None, None)
                overlay = raster_viz.raster_overlay_layer(
                    mean_arr, raster["transform"], raster["crs"], METRIC_LABELS.get(metric, metric), colorscale,
                    vmin=raster_vmin, vmax=raster_vmax,
                )
                if overlay is None:
                    st.info("No finite pixel values to display for this metric/sensor.")
                else:
                    data_uri, bounds, legend, vmin, vmax = overlay
                    raster_map = folium.Map(tiles=None, control_scale=True)
                    satellite_tile_layer().add_to(raster_map)
                    folium.raster_layers.ImageOverlay(
                        image=data_uri, bounds=bounds, opacity=1.0, interactive=False, cross_origin=False,
                        name=f"{METRIC_LABELS.get(metric, metric)} mean",
                    ).add_to(raster_map)
                    outline_gdf = gpd.GeoDataFrame(geometry=[sel.geometry], crs=polygons.WGS84)
                    folium.GeoJson(
                        outline_gdf,
                        style_function=lambda f: {"color": "#FFFFFF", "weight": 3, "fillOpacity": 0, "dashArray": "5,5"},
                        name="Selected polygon",
                    ).add_to(raster_map)
                    legend.add_to(raster_map)
                    raster_map.fit_bounds(bounds)
                    fix_map_visibility_zoom(raster_map, bounds)
                    st_folium(
                        raster_map,
                        key=f"raster_map_{sel['uid']}_{raster_product}_{metric}_{len(kept_df)}",
                        height=550, use_container_width=True, returned_objects=[],
                    )
                    st.caption(
                        f"{PRODUCT_LABELS.get(raster_product, raster_product)} - "
                        f"{int(np.nanmax(n_scenes_arr)) if n_scenes_arr.size else 0} scenes max contributing per pixel. "
                        "Dashed white outline is the selected polygon (original, unsimplified boundary)."
                    )

                try:
                    import rasterio

                    tmp_path = os.path.join(tempfile.mkdtemp(prefix="peatland_raster_"), "raster.tif")
                    out = np.where(np.isnan(mean_arr), -9999.0, mean_arr).astype("float32")
                    profile = dict(
                        driver="GTiff", height=out.shape[0], width=out.shape[1], count=1,
                        dtype="float32", crs=raster["crs"], transform=raster["transform"],
                        nodata=-9999.0, compress="lzw",
                    )
                    with rasterio.open(tmp_path, "w", **profile) as dst:
                        dst.write(out, 1)
                    with open(tmp_path, "rb") as f:
                        st.download_button(
                            "Download this raster as GeoTIFF",
                            f.read(),
                            file_name=f"{sel['uid']}_{raster_product}_{metric}_mean.tif",
                            mime="image/tiff",
                        )
                except Exception as e:
                    st.caption(f"(GeoTIFF export unavailable: {e})")

        # ---- Time series --------------------------------------------------
        st.divider()
        st.subheader(f"{METRIC_LABELS.get(metric, metric)} over time")
        ts_all = kept_df.dropna(subset=[mean_col]).copy()
        ts_all["datetime"] = pd.to_datetime(ts_all["datetime"])
        ts_all["month"] = ts_all["datetime"].dt.month
        month_choice = st.multiselect(
            "Months to include", options=list(range(1, 13)),
            default=list(range(1, 13)), format_func=lambda m: calendar.month_name[m],
            key="ts_month_choice",
        )
        trend_c1, trend_c2 = st.columns(2)
        with trend_c1:
            linear_on = st.radio("Linear trend", ["Off", "On"], horizontal=True, key="ts_linear") == "On"
        with trend_c2:
            gam_on = st.radio("GAM (smoothed) trend", ["Off", "On"], horizontal=True, key="ts_gam") == "On"

        if not month_choice:
            st.info("Select at least one month above.")
            ts = ts_all.iloc[0:0]
        else:
            ts = ts_all[ts_all["month"].isin(month_choice)]

        if len(ts) == 0:
            st.info("No scenes fall in the selected month(s)/filters.")
        else:
            fig_ts = go.Figure()
            has_multi_sensor = ts["sensor"].nunique() > 1
            for mo in sorted(ts["month"].unique()):
                sub = ts[ts["month"] == mo].sort_values("datetime")
                symbols = [SENSOR_SYMBOLS.get(s, "circle") for s in sub["sensor"]]
                fig_ts.add_trace(
                    go.Scatter(
                        x=sub["datetime"], y=sub[mean_col],
                        error_y=dict(type="data", array=sub[std_col], visible=True, thickness=1),
                        mode="markers", name=calendar.month_name[mo],
                        marker=dict(size=8, color=MONTH_COLORS[mo], symbol=symbols,
                                    line=dict(width=1, color="rgba(0,0,0,0.4)")),
                        customdata=sub[["item_id", "product", "date", "sensor"]].values,
                        hovertemplate="%{customdata[3]} %{customdata[2]}<br>Value: %{y:.3f}<extra></extra>",
                    )
                )
            if has_multi_sensor:
                st.caption("Marker shape: ● Landsat, ◆ Sentinel-2. Colour = month.")

            x_days = trends.days_since(ts["datetime"])
            if linear_on:
                lt = trends.linear_trend(x_days, ts[mean_col].values)
                if lt is not None:
                    fig_ts.add_trace(
                        go.Scatter(
                            x=ts["datetime"].values[np.argsort(x_days)], y=lt.y,
                            mode="lines", name=f"Linear trend ({lt.slope_per_year:+.4f}/yr, R²={lt.r2:.2f})",
                            line=dict(color="black", dash="dash"),
                        )
                    )
                else:
                    st.caption("Not enough points for a linear trend.")
            if gam_on:
                x_grid_days, y_grid, gam_error = trends.gam_trend(x_days, ts[mean_col].values)
                if gam_error is None:
                    x_grid_dates = ts["datetime"].min() + pd.to_timedelta(x_grid_days, unit="D")
                    fig_ts.add_trace(
                        go.Scatter(
                            x=x_grid_dates, y=y_grid, mode="lines", name="GAM smoothed trend",
                            line=dict(color="#8E44AD", width=3),
                        )
                    )
                else:
                    st.caption(f"Could not fit a GAM trend: {gam_error}")

            if metric == "TCW":
                fig_ts.add_hline(
                    y=0, line_dash="dot", line_color="#1B4F72", line_width=1.5,
                    annotation_text="0 - possible open water", annotation_position="bottom right",
                    annotation_font_color="#1B4F72",
                )
            fig_ts.update_layout(
                yaxis_title=METRIC_LABELS.get(metric, metric), xaxis_title="Date",
                height=520, legend_title="Month",
            )
            ts_event = st.plotly_chart(
                fig_ts, use_container_width=True, on_select="rerun",
                selection_mode=("points", "box", "lasso"), key="ts_chart",
            )

            sel_points = []
            if ts_event and ts_event.get("selection"):
                sel_points = ts_event["selection"].get("points", [])
            sel_scenes = []
            for pt in sel_points:
                cd = pt.get("customdata")
                if cd and len(cd) >= 4:
                    sel_scenes.append(dict(item_id=cd[0], product=cd[1], date=cd[2], sensor=cd[3]))

            if sel_scenes:
                st.markdown(
                    f"**{len(sel_scenes)} scene(s) selected:** "
                    + ", ".join(f"{d['sensor']} {d['date']}" for d in sel_scenes[:10])
                    + (" ..." if len(sel_scenes) > 10 else "")
                )
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    view_label = f"Add selected scene(s) to viewer (up to {MAX_VIEWED_SCENES} at once)"
                    if st.button(view_label):
                        viewed_keys = {v["key"] for v in st.session_state.viewed_scenes}
                        to_fetch = [
                            d for d in sel_scenes
                            if analysis.scene_key(d["product"], d["item_id"]) not in viewed_keys
                        ]
                        n_dupe = len(sel_scenes) - len(to_fetch)
                        n_slots_left = max(MAX_VIEWED_SCENES - len(st.session_state.viewed_scenes), 0)
                        n_over_cap = max(len(to_fetch) - n_slots_left, 0)
                        to_fetch = to_fetch[:n_slots_left]

                        n_fetched, fetch_errors = 0, []
                        for d in to_fetch:
                            key = analysis.scene_key(d["product"], d["item_id"])
                            try:
                                with st.spinner(f"Fetching {d['sensor']} {d['date']}..."):
                                    data = analysis.fetch_single_scene(result, d["product"], d["item_id"])
                                st.session_state.viewed_scenes.append(
                                    dict(key=key, sensor=d["sensor"], date=d["date"], data=data)
                                )
                                n_fetched += 1
                            except Exception as e:
                                fetch_errors.append(f"{d['sensor']} {d['date']}: {e}")

                        if n_over_cap:
                            st.warning(
                                f"The viewer already holds {MAX_VIEWED_SCENES} scenes (its max) - "
                                f"{n_over_cap} of the selected scene(s) weren't added. Remove one "
                                "below first to make room."
                            )
                        if n_dupe:
                            st.caption(f"{n_dupe} selected scene(s) were already in the viewer.")
                        for msg in fetch_errors:
                            st.error(f"Could not fetch {msg}")
                        if n_fetched:
                            st.rerun()
                with bcol2:
                    sel_keys = {analysis.scene_key(d["product"], d["item_id"]) for d in sel_scenes}
                    all_excluded = sel_keys.issubset(st.session_state.excluded_keys)
                    label = "Re-include selected scene(s)" if all_excluded else "Exclude selected scene(s)"
                    if st.button(label):
                        if all_excluded:
                            st.session_state.excluded_keys -= sel_keys
                        else:
                            st.session_state.excluded_keys |= sel_keys
                        st.rerun()

        # ---- Multi-scene raster viewer ---------------------------------------
        # Up to MAX_VIEWED_SCENES scenes' rasters in memory here at once -
        # fetch_single_scene() re-fetches directly from DEA on demand (nothing
        # pre-cached for this). Remove a scene below to free it and make room
        # for another; download anything you want to keep, since it's gone
        # once removed or once a new analysis run replaces the whole list.
        viewed_scenes = st.session_state.viewed_scenes
        if viewed_scenes:
            st.divider()
            st.subheader(f"Inspect scene(s) ({len(viewed_scenes)}/{MAX_VIEWED_SCENES})")

            for v in viewed_scenes:
                vrow1, vrow2 = st.columns([5, 1])
                vrow1.caption(f"{v['sensor']} {v['date']} ({v['key'].split('::', 1)[-1]})")
                if vrow2.button("Remove", key=f"remove_scene_{v['key']}"):
                    st.session_state.viewed_scenes = [
                        x for x in st.session_state.viewed_scenes if x["key"] != v["key"]
                    ]
                    st.rerun()
            if len(viewed_scenes) > 1 and st.button("Clear all"):
                st.session_state.viewed_scenes = []
                st.rerun()

            vmode_col, voutline_col = st.columns([3, 1])
            with vmode_col:
                view_mode = st.radio(
                    "Display", ["True colour", f"{METRIC_LABELS.get(metric, metric)} layer"],
                    horizontal=True, key="scene_view_mode",
                )
            with voutline_col:
                show_outline = st.checkbox("Show polygon outline", value=True, key="scene_show_outline")
            rshow_col, ropacity_col = st.columns([1, 2])
            with rshow_col:
                show_raster = st.checkbox("Show raster(s)", value=True, key="scene_show_raster")
            with ropacity_col:
                raster_opacity = st.slider(
                    "Raster opacity", min_value=0, max_value=100, value=85,
                    key="scene_raster_opacity", disabled=not show_raster,
                )
            st.caption(
                f"Each cropped to a square {analysis.SCENE_VIEW_BUFFER_M:.0f} m beyond the polygon on "
                "every side, for landscape context - not just the polygon itself. With more than one "
                "scene shown, use the layer control (top right of the map) to toggle individual scenes."
            )

            scene_map = folium.Map(tiles=None, control_scale=True)
            satellite_tile_layer().add_to(scene_map)

            all_bounds = []
            downloads = []  # (label, array, kind, transform, crs, key)
            for v in viewed_scenes:
                cached = v["data"]
                vlabel = f"{v['sensor']} {v['date']}"
                bounds = None
                legend = None
                download_arr = None
                download_kind = None
                try:
                    if view_mode == "True colour":
                        if cached["rgb"] is not None:
                            layer_out = raster_viz.rgb_overlay_layer(
                                cached["rgb"], cached["transform"], cached["crs"], opacity=raster_opacity / 100.0
                            )
                            if layer_out is not None:
                                data_uri, bounds = layer_out
                                download_arr, download_kind = cached["rgb"], "rgb"
                    else:
                        arr = cached["metrics"].get(metric)
                        if arr is not None:
                            scene_vmin, scene_vmax = (-1.0, 1.0) if metric == "NDVI" else (None, None)
                            layer_out = raster_viz.raster_overlay_layer(
                                arr, cached["transform"], cached["crs"], METRIC_LABELS.get(metric, metric),
                                colorscale, vmin=scene_vmin, vmax=scene_vmax, opacity=raster_opacity / 100.0,
                            )
                            if layer_out is not None:
                                data_uri, bounds, legend, _vmin, _vmax = layer_out
                                download_arr, download_kind = arr, "single"
                except Exception as e:
                    st.caption(f"Could not render {vlabel}: {e}")

                if bounds is not None:
                    all_bounds.append(bounds)
                    if show_raster:
                        folium.raster_layers.ImageOverlay(
                            image=data_uri, bounds=bounds, opacity=1.0, interactive=False,
                            cross_origin=False, name=vlabel,
                        ).add_to(scene_map)
                        if legend is not None:
                            legend.add_to(scene_map)
                    downloads.append(
                        (vlabel, download_arr, download_kind, cached["transform"], cached["crs"], v["key"])
                    )

            if show_outline:
                outline_gdf = gpd.GeoDataFrame(geometry=[sel.geometry], crs=polygons.WGS84)
                folium.GeoJson(
                    outline_gdf,
                    style_function=lambda f: {"color": "#FFFFFF", "weight": 3, "fillOpacity": 0, "dashArray": "5,5"},
                    name="Selected polygon",
                ).add_to(scene_map)

            if not all_bounds:
                st.info("Nothing to display for the current view mode.")
            else:
                if len(viewed_scenes) > 1:
                    folium.LayerControl(collapsed=False).add_to(scene_map)
                union_bounds = [
                    [min(b[0][0] for b in all_bounds), min(b[0][1] for b in all_bounds)],
                    [max(b[1][0] for b in all_bounds), max(b[1][1] for b in all_bounds)],
                ]
                scene_map.fit_bounds(union_bounds)
                fix_map_visibility_zoom(scene_map, union_bounds)
                map_key_part = "_".join(v["key"] for v in viewed_scenes)
                st_folium(
                    scene_map,
                    key=f"scene_view_map_{map_key_part}_{view_mode}_{show_raster}_{raster_opacity}",
                    height=550, use_container_width=True, returned_objects=[],
                )

                for vlabel, download_arr, download_kind, transform, crs, key in downloads:
                    if download_arr is None:
                        continue
                    try:
                        import rasterio

                        tmp_path = os.path.join(tempfile.mkdtemp(prefix="peatland_scene_"), "scene.tif")
                        if download_kind == "rgb":
                            out = np.where(np.isnan(download_arr), -9999.0, download_arr).astype("float32")
                            out = np.moveaxis(out, -1, 0)  # (h,w,bands) -> (bands,h,w) for rasterio
                            profile = dict(
                                driver="GTiff", height=out.shape[1], width=out.shape[2], count=out.shape[0],
                                dtype="float32", crs=crs, transform=transform,
                                nodata=-9999.0, compress="lzw",
                            )
                            with rasterio.open(tmp_path, "w", **profile) as dst:
                                dst.write(out)
                            dl_label, dl_suffix = "true_colour", "rgb"
                        else:
                            out = np.where(np.isnan(download_arr), -9999.0, download_arr).astype("float32")
                            profile = dict(
                                driver="GTiff", height=out.shape[0], width=out.shape[1], count=1,
                                dtype="float32", crs=crs, transform=transform,
                                nodata=-9999.0, compress="lzw",
                            )
                            with rasterio.open(tmp_path, "w", **profile) as dst:
                                dst.write(out, 1)
                            dl_label, dl_suffix = metric, metric
                        with open(tmp_path, "rb") as f:
                            st.download_button(
                                f"Download {vlabel} ({dl_label}) as GeoTIFF",
                                f.read(),
                                file_name=f"{key.replace('::', '_')}_{dl_suffix}.tif",
                                mime="image/tiff",
                                key=f"download_{key}_{dl_suffix}",
                            )
                    except Exception as e:
                        st.caption(f"(GeoTIFF export unavailable for {vlabel}: {e})")

        # ---- Phenology ------------------------------------------------------
        st.divider()
        st.subheader("Phenology - mean value by month")
        st.caption(
            "Every scene passing the quality filters above, across all years, plotted by month, "
            "running July -> June to keep the (Southern Hemisphere) growing season together in one "
            "block - shows the within-year seasonal pattern independent of any long-term trend."
        )
        phen_src = kept_df.dropna(subset=[mean_col]).copy()
        if len(phen_src) == 0:
            st.info("No scenes available for the phenology plot after quality filtering.")
        else:
            phen_src["datetime"] = pd.to_datetime(phen_src["datetime"])
            phen_src["month"] = phen_src["datetime"].dt.month
            phen_src["water_month"] = phen_src["month"].apply(to_water_month)
            phen_src["year"] = phen_src["datetime"].dt.year
            phen_src["month_abbr"] = phen_src["month"].apply(lambda mo: calendar.month_abbr[mo])

            phen_trend_c1, phen_trend_c2 = st.columns(2)
            with phen_trend_c1:
                phen_linear_on = st.radio("Linear trend", ["Off", "On"], horizontal=True, key="phen_linear") == "On"
            with phen_trend_c2:
                phen_gam_on = st.radio("GAM (smoothed) trend", ["Off", "On"], horizontal=True, key="phen_gam") == "On"

            fig_phen = go.Figure()
            for s, sub in phen_src.groupby("sensor"):
                fig_phen.add_trace(
                    go.Scatter(
                        x=sub["water_month"], y=sub[mean_col], mode="markers", name=s,
                        marker=dict(size=7, opacity=0.6, symbol=SENSOR_SYMBOLS.get(s, "circle")),
                        customdata=sub[["year", "month_abbr"]].values,
                        hovertemplate="%{customdata[1]} %{customdata[0]}<br>Value %{y:.3f}<extra></extra>",
                    )
                )
            monthly_mean = phen_src.groupby("water_month")[mean_col].mean().reindex(range(1, 13))
            fig_phen.add_trace(
                go.Scatter(
                    x=monthly_mean.index, y=monthly_mean.values, mode="lines+markers",
                    name="Monthly mean (all sensors/years)",
                    line=dict(color="black", width=2), marker=dict(size=8, symbol="diamond"),
                )
            )

            x_month = phen_src["water_month"].astype(float).values
            y_val = phen_src[mean_col].values
            if phen_linear_on:
                lt = trends.linear_trend(x_month, y_val)
                if lt is not None:
                    fig_phen.add_trace(
                        go.Scatter(
                            x=lt.x, y=lt.y, mode="lines", name="Linear trend",
                            line=dict(color="#2C3E50", dash="dash", width=2),
                        )
                    )
                    mid_i = len(lt.x) // 2
                    fig_phen.add_annotation(
                        x=lt.x[mid_i], y=lt.y[mid_i],
                        text=f"Slope = {lt.slope_per_day:+.4f} / month (R²={lt.r2:.2f})",
                        showarrow=True, arrowhead=1, ax=0, ay=-35,
                        font=dict(color="#2C3E50", size=12),
                    )
                else:
                    st.caption("Not enough points for a linear trend.")
            if phen_gam_on:
                x_grid, y_grid, gam_error = trends.gam_trend(x_month, y_val)
                if gam_error is None:
                    fig_phen.add_trace(
                        go.Scatter(
                            x=x_grid, y=y_grid, mode="lines", name="GAM smoothed trend",
                            line=dict(color="#8E44AD", width=3),
                        )
                    )
                else:
                    st.caption(f"Could not fit a GAM trend: {gam_error}")

            if metric == "TCW":
                fig_phen.add_hline(
                    y=0, line_dash="dot", line_color="#1B4F72", line_width=1.5,
                    annotation_text="0 - possible open water", annotation_position="bottom right",
                    annotation_font_color="#1B4F72",
                )
            fig_phen.update_layout(
                xaxis=dict(
                    title="Month", tickmode="array", tickvals=list(range(1, 13)),
                    ticktext=[calendar.month_abbr[m] for m in WATER_MONTH_ORDER],
                ),
                yaxis_title=METRIC_LABELS.get(metric, metric), height=520, legend_title="Sensor",
            )
            st.plotly_chart(fig_phen, use_container_width=True)

        # ---- Per-scene table --------------------------------------------------
        st.divider()
        st.subheader("Per-scene data")
        hide_empty = st.checkbox(
            "Hide scenes with no valid pixels", value=True,
            help="Hides scenes where cloud/shadow/snow/no-data masking left zero usable pixels over "
            "the polygon.",
        )
        sensor_filter = st.multiselect(
            "Filter by sensor", sorted(df_all["sensor"].unique()), default=sorted(df_all["sensor"].unique())
        )
        table_df = df_all[df_all["sensor"].isin(sensor_filter)] if sensor_filter else df_all.copy()
        if hide_empty:
            table_df = table_df[table_df["n_valid_px"] > 0]
        table_df = table_df.reset_index(drop=True)
        table_keys = table_df["product"] + "::" + table_df["item_id"].astype(str)
        table_df.insert(0, "excluded", table_keys.isin(st.session_state.excluded_keys).values)

        st.caption(
            f"Showing {len(table_df)} of {len(df_all)} fetched scenes. Tick/untick 'excluded' to "
            "manually include or exclude a scene from the raster/plots above (e.g. an anomaly the "
            "smoke filter didn't catch)."
        )
        edited_df = st.data_editor(
            table_df,
            use_container_width=True, height=400, key="scene_table_editor", hide_index=True,
            disabled=[c for c in table_df.columns if c != "excluded"],
            column_config={"excluded": st.column_config.CheckboxColumn("excluded")},
        )
        new_excluded = set(table_keys[edited_df["excluded"].values])
        new_included = set(table_keys[~edited_df["excluded"].values])
        updated = (st.session_state.excluded_keys - new_included) | new_excluded
        if updated != st.session_state.excluded_keys:
            st.session_state.excluded_keys = updated
            st.rerun()

        st.download_button(
            "Download per-scene data as CSV",
            table_df.drop(columns=["excluded"]).to_csv(index=False).encode("utf-8"),
            file_name=f"{sel['uid']}_peatland_trend_data.csv",
            mime="text/csv",
        )

# ==========================================================================
# Tab 4 - trend comparisons across uploaded CSVs
# ==========================================================================
with tab_compare:
    st.subheader("Compare trends across multiple exported CSVs")
    st.write(
        "Upload two or more per-scene data CSVs (exported from tab 3) to plot their trends side by "
        "side, one facet per file - handy for comparing different polygons, or different runs of the "
        "same polygon."
    )
    uploaded = st.file_uploader(
        "Upload per-scene data CSV(s)", type="csv", accept_multiple_files=True, key="compare_uploader"
    )

    if not uploaded:
        st.info("Upload CSV files exported from tab 3's 'Download per-scene data as CSV' button.")
    else:
        frames = []
        problems = []
        for f in uploaded:
            try:
                d = pd.read_csv(f)
                if "datetime" not in d.columns:
                    problems.append(f"{f.name}: no 'datetime' column - skipped.")
                    continue
                d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
                d = d.dropna(subset=["datetime"])
                dataset_label = os.path.splitext(f.name)[0]
                d["dataset"] = dataset_label
                frames.append(d)
            except Exception as e:
                problems.append(f"{f.name}: failed to read - {e}")

        for p in problems:
            st.warning(p)

        if not frames:
            st.error("None of the uploaded files could be read.")
        else:
            combined = pd.concat(frames, ignore_index=True, sort=False)
            combined["month"] = combined["datetime"].dt.month
            combined["month_name"] = combined["month"].apply(lambda m: calendar.month_name[m])
            available_metrics = [m for m in METRICS if f"{m}_mean" in combined.columns]
            if not available_metrics:
                st.error(
                    "None of the uploaded files have a '<metric>_mean' column - are these per-scene "
                    "data CSVs exported from tab 3?"
                )
            else:
                # ---- Filters (same quality gate as tab 3, applied across
                # every uploaded dataset before plotting) --------------------
                st.subheader("Filters")
                st.caption(
                    "Same filters as tab 3, applied here to every uploaded dataset. Any column a "
                    "given CSV doesn't have (e.g. an older export, or a hand-edited file) is simply "
                    "skipped for that check rather than excluding everything."
                )
                cmp_month_choice = st.multiselect(
                    "Months to include", options=list(range(1, 13)),
                    default=list(range(1, 13)), format_func=lambda m: calendar.month_name[m],
                    key="cmp_month_choice",
                )
                fq1, fq2 = st.columns(2)
                with fq1:
                    cmp_min_valid = st.slider(
                        "Minimum valid pixels per scene (%)", 0, 100, 50, key="cmp_min_valid"
                    )
                    cmp_max_snow = st.slider(
                        "Maximum scene snow cover (%)", 0, 100, 100, key="cmp_max_snow"
                    )
                with fq2:
                    cmp_max_cloud = st.slider(
                        "Maximum scene cloud cover over polygon (%)", 0, 100, 100, key="cmp_max_cloud"
                    )
                    cmp_max_smoke = st.slider(
                        "Maximum scene smoke/haze cover (%)", 0, 100, 100, key="cmp_max_smoke"
                    )

                if not cmp_month_choice:
                    st.info("Select at least one month above.")
                    filtered = combined.iloc[0:0]
                else:
                    filtered = analysis.filter_uploaded_df(
                        combined, cmp_min_valid, cmp_max_snow, cmp_max_cloud, cmp_max_smoke,
                        months=cmp_month_choice,
                    )
                st.caption(f"{len(filtered)} of {len(combined)} uploaded rows currently pass these filters.")

                st.divider()
                color_col_by_label = {"sensor": "sensor", "dataset": "dataset", "month": "month_name"}
                color_labels = [
                    label for label, col in color_col_by_label.items() if col in combined.columns
                ] or ["dataset"]
                cmp_c1, cmp_c2, cmp_c3 = st.columns(3)
                with cmp_c1:
                    cmp_metric = st.selectbox(
                        "Metric", available_metrics, format_func=lambda m: METRIC_LABELS.get(m, m),
                        key="cmp_metric_select",
                    )
                with cmp_c2:
                    facet_cols_input = st.number_input(
                        "Facets per row", min_value=1, max_value=6, value=3, key="cmp_facet_cols"
                    )
                with cmp_c3:
                    color_by_label = st.selectbox("Colour points by", color_labels, key="cmp_color_by")
                color_by = color_col_by_label[color_by_label]

                cmp_mean_col = f"{cmp_metric}_mean"
                cmp_std_col = f"{cmp_metric}_std"
                plot_df = filtered.dropna(subset=[cmp_mean_col]).sort_values("datetime")
                if cmp_std_col not in plot_df.columns:
                    plot_df[cmp_std_col] = None

                if len(plot_df) == 0:
                    st.info(f"No rows have a value for {METRIC_LABELS.get(cmp_metric, cmp_metric)} after filtering.")
                else:
                    dataset_order = sorted(plot_df["dataset"].unique().tolist())
                    n_cols = max(1, min(int(facet_cols_input), len(dataset_order)))
                    n_rows = -(-len(dataset_order) // n_cols)
                    metric_label = METRIC_LABELS.get(cmp_metric, cmp_metric)

                    if color_by == "month_name":
                        month_order = [calendar.month_name[m] for m in range(1, 13)]
                        color_map = {calendar.month_name[m]: MONTH_COLORS[m] for m in range(1, 13)}
                    else:
                        color_map = build_categorical_color_map(plot_df[color_by])

                    # ---- Long-term trends (faceted, one subplot per dataset) ----
                    st.markdown("**Long-term trends**")
                    trend_c1, trend_c2 = st.columns(2)
                    with trend_c1:
                        cmp_linear_on = st.radio(
                            "Linear trend", ["Off", "On"], horizontal=True, key="cmp_linear"
                        ) == "On"
                    with trend_c2:
                        cmp_gam_on = st.radio(
                            "GAM (smoothed) trend", ["Off", "On"], horizontal=True, key="cmp_gam"
                        ) == "On"
                    st.caption(
                        "Trend lines are fit separately per dataset (per facet), from that dataset's own "
                        "points only - a single line spanning multiple different polygons/runs wouldn't "
                        "mean anything."
                    )

                    fig_trends = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=dataset_order)
                    linear_fails, gam_fails = [], []
                    for i, ds in enumerate(dataset_order):
                        r, c = facet_rc(i, n_cols)
                        sub_ds = plot_df[plot_df["dataset"] == ds]
                        group_col = color_by if color_by in sub_ds.columns else None
                        groups = sub_ds.groupby(group_col) if group_col else [(ds, sub_ds)]
                        for name, g in groups:
                            fig_trends.add_trace(
                                go.Scatter(
                                    x=g["datetime"], y=g[cmp_mean_col],
                                    error_y=dict(type="data", array=g[cmp_std_col], visible=True, thickness=1),
                                    mode="markers", name=str(name), legendgroup=str(name), showlegend=(i == 0),
                                    marker=dict(size=6, color=color_map.get(name)),
                                ),
                                row=r, col=c,
                            )
                        x_days = trends.days_since(sub_ds["datetime"])
                        dt_min = sub_ds["datetime"].min()
                        if cmp_linear_on:
                            lt = trends.linear_trend(x_days, sub_ds[cmp_mean_col].values)
                            if lt is not None:
                                x_dates = dt_min + pd.to_timedelta(lt.x, unit="D")
                                fig_trends.add_trace(
                                    go.Scatter(
                                        x=x_dates, y=lt.y, mode="lines",
                                        name=f"Linear trend ({lt.slope_per_year:+.4f}/yr)",
                                        legendgroup="linear_trend", showlegend=(i == 0),
                                        line=dict(color="black", dash="dash"),
                                    ),
                                    row=r, col=c,
                                )
                            else:
                                linear_fails.append(ds)
                        if cmp_gam_on:
                            xg, yg, err = trends.gam_trend(x_days, sub_ds[cmp_mean_col].values)
                            if err is None:
                                x_grid_dates = dt_min + pd.to_timedelta(xg, unit="D")
                                fig_trends.add_trace(
                                    go.Scatter(
                                        x=x_grid_dates, y=yg, mode="lines", name="GAM smoothed trend",
                                        legendgroup="gam_trend", showlegend=(i == 0),
                                        line=dict(color="#8E44AD", width=3),
                                    ),
                                    row=r, col=c,
                                )
                            else:
                                gam_fails.append(f"{ds} ({err})")

                    if cmp_metric == "TCW":
                        fig_trends.add_hline(
                            y=0, line_dash="dot", line_color="#1B4F72", line_width=1.5, row="all", col="all",
                        )
                    # matches="x"/"y" (rather than None) links every facet's axis to the
                    # first one's, so they all zoom/pan together and share one range -
                    # makes magnitudes comparable at a glance across datasets, instead of
                    # each facet silently autoscaling to its own min/max.
                    fig_trends.update_yaxes(matches="y", showticklabels=True, title_text=metric_label)
                    fig_trends.update_xaxes(matches="x")
                    fig_trends.update_layout(height=350 * n_rows)
                    st.plotly_chart(fig_trends, use_container_width=True)
                    if linear_fails:
                        st.caption(f"Not enough points for a linear trend: {', '.join(linear_fails)}.")
                    if gam_fails:
                        st.caption(f"Could not fit a GAM trend: {'; '.join(gam_fails)}.")

                    # ---- Phenology (faceted, one subplot per dataset) ------------
                    st.divider()
                    st.markdown("**Phenology - mean value by month**")
                    st.caption(
                        "Every point across all years, by month, running July -> June (Southern "
                        "Hemisphere growing season), one facet per dataset - same as tab 3's phenology "
                        "plot, fit and drawn separately per dataset."
                    )
                    phen_trend_c1, phen_trend_c2 = st.columns(2)
                    with phen_trend_c1:
                        cmp_phen_linear_on = st.radio(
                            "Linear trend", ["Off", "On"], horizontal=True, key="cmp_phen_linear"
                        ) == "On"
                    with phen_trend_c2:
                        cmp_phen_gam_on = st.radio(
                            "GAM (smoothed) trend", ["Off", "On"], horizontal=True, key="cmp_phen_gam"
                        ) == "On"

                    phen_df = plot_df.copy()
                    phen_df["water_month"] = phen_df["month"].apply(to_water_month)
                    phen_df["year"] = phen_df["datetime"].dt.year
                    phen_df["month_abbr"] = phen_df["month"].apply(lambda mo: calendar.month_abbr[mo])
                    sensor_color_map = (
                        build_categorical_color_map(phen_df["sensor"]) if "sensor" in phen_df.columns else {}
                    )

                    fig_phen_cmp = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=dataset_order)
                    phen_linear_fails, phen_gam_fails = [], []
                    for i, ds in enumerate(dataset_order):
                        r, c = facet_rc(i, n_cols)
                        sub_ds = phen_df[phen_df["dataset"] == ds]
                        sensor_groups = (
                            sub_ds.groupby("sensor") if "sensor" in sub_ds.columns else [(ds, sub_ds)]
                        )
                        for sensor_val, g in sensor_groups:
                            fig_phen_cmp.add_trace(
                                go.Scatter(
                                    x=g["water_month"], y=g[cmp_mean_col], mode="markers", name=str(sensor_val),
                                    legendgroup=str(sensor_val), showlegend=(i == 0),
                                    marker=dict(
                                        size=6, opacity=0.65, symbol=SENSOR_SYMBOLS.get(sensor_val, "circle"),
                                        color=sensor_color_map.get(sensor_val),
                                    ),
                                    customdata=g[["year", "month_abbr"]].values,
                                    hovertemplate="%{customdata[1]} %{customdata[0]}<br>Value %{y:.3f}<extra></extra>",
                                ),
                                row=r, col=c,
                            )
                        monthly_mean = sub_ds.groupby("water_month")[cmp_mean_col].mean().reindex(range(1, 13))
                        fig_phen_cmp.add_trace(
                            go.Scatter(
                                x=monthly_mean.index, y=monthly_mean.values, mode="lines+markers",
                                name="Monthly mean", legendgroup="monthly_mean", showlegend=(i == 0),
                                line=dict(color="black", width=2), marker=dict(size=6, symbol="diamond"),
                            ),
                            row=r, col=c,
                        )
                        x_month = sub_ds["water_month"].astype(float).values
                        y_val = sub_ds[cmp_mean_col].values
                        if cmp_phen_linear_on:
                            lt = trends.linear_trend(x_month, y_val)
                            if lt is not None:
                                fig_phen_cmp.add_trace(
                                    go.Scatter(
                                        x=lt.x, y=lt.y, mode="lines", name="Linear trend",
                                        legendgroup="phen_linear_trend", showlegend=(i == 0),
                                        line=dict(color="#2C3E50", dash="dash", width=2),
                                    ),
                                    row=r, col=c,
                                )
                                mid_i = len(lt.x) // 2
                                fig_phen_cmp.add_annotation(
                                    x=lt.x[mid_i], y=lt.y[mid_i], text=f"{lt.slope_per_day:+.4f}/mo",
                                    showarrow=True, arrowhead=1, ax=0, ay=-25,
                                    font=dict(color="#2C3E50", size=10), row=r, col=c,
                                )
                            else:
                                phen_linear_fails.append(ds)
                        if cmp_phen_gam_on:
                            xg, yg, err = trends.gam_trend(x_month, y_val)
                            if err is None:
                                fig_phen_cmp.add_trace(
                                    go.Scatter(
                                        x=xg, y=yg, mode="lines", name="GAM smoothed trend",
                                        legendgroup="phen_gam_trend", showlegend=(i == 0),
                                        line=dict(color="#8E44AD", width=3),
                                    ),
                                    row=r, col=c,
                                )
                            else:
                                phen_gam_fails.append(f"{ds} ({err})")

                    if cmp_metric == "TCW":
                        fig_phen_cmp.add_hline(
                            y=0, line_dash="dot", line_color="#1B4F72", line_width=1.5, row="all", col="all",
                        )
                    # Shared axes across facets here too - see the same note on fig_trends
                    # above.
                    fig_phen_cmp.update_xaxes(
                        tickmode="array", tickvals=list(range(1, 13)),
                        ticktext=[calendar.month_abbr[m] for m in WATER_MONTH_ORDER], title_text="Month",
                        matches="x",
                    )
                    fig_phen_cmp.update_yaxes(matches="y", showticklabels=True, title_text=metric_label)
                    fig_phen_cmp.update_layout(height=350 * n_rows)
                    st.plotly_chart(fig_phen_cmp, use_container_width=True)
                    if phen_linear_fails:
                        st.caption(f"Not enough points for a linear trend: {', '.join(phen_linear_fails)}.")
                    if phen_gam_fails:
                        st.caption(f"Could not fit a GAM trend: {'; '.join(phen_gam_fails)}.")

                st.divider()
                st.subheader("Filtered data")
                st.dataframe(plot_df, use_container_width=True, height=350)
                st.download_button(
                    "Download filtered data as CSV",
                    plot_df.to_csv(index=False).encode("utf-8"),
                    file_name="trend_comparison_combined.csv",
                    mime="text/csv",
                )
