# Peatland Trend Analysis

A Streamlit app for analysing long-term trends in Victorian alpine peatlands
and Snowy Mountains (NSW) peatlands/fens, using Landsat and Sentinel-2
surface reflectance from Digital Earth Australia (DEA). Pick a polygon on
a map, choose a date range and sensor(s), fetch, then interactively filter,
trend-plot, spot-check and export.

It's a Python/Streamlit rebuild of an original R Shiny app, fetching data
live from DEA rather than relying on local pre-extracted CSVs.

## Setup (one-off)

```
cd Peatland_trends
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Double-click `run_app.bat`, or from an activated environment:

```
cd Peatland_trends
streamlit run app.py
```

This runs entirely locally except for the STAC/raster requests to DEA's own
servers. DEA access is anonymous and public - no personal API key, account,
or credential is needed anywhere in this pipeline.

## How to use it

The app opens on a **Home** tab with a plain-language overview of what it
does and a short walkthrough of each numbered tab, plus a "Good to know"
section covering the gotchas below (cloud cover, smoke detection, the
water-year phenology axis, the TCW zero line, NDVI's fixed legend, and the
scene viewer's 5-scene cap) - worth reading before your first run,
otherwise everything past this point in the README covers the same ground
in more detail.

1. **Tab 1 - Select polygon.** Click a polygon on the map (highlighted in
   red once selected), or use the searchable dropdown in the sidebar - with
   ~12,000 small polygons across both layers combined, the search box is
   often faster and more precise than clicking. The two layers can be
   toggled independently in the sidebar. With nothing selected the map
   frames the combined extent of both polygon layers (not a world view);
   once you select a polygon it zooms to a padded box around that polygon
   specifically.
2. **Tab 2 - Configure & run.** Controls here only affect what gets
   *fetched* from DEA:
   - Date range, bounded by Landsat 5's launch (1 March 1984) and today.
   - Tick Landsat, Sentinel-2, or both.
   - *Advanced fetch options*: a whole-tile cloud-cover ceiling (DEA's
     `eo:cloud_cover`, computed over the entire 100km+ tile, not your
     polygon - see "Why does 0% cloud return nothing?" below), a minimum
     tile-coverage percentage (only scenes whose tile fully/mostly overlaps
     the polygon are fetched at all), a smoke/haze detection sensitivity,
     and a batch size (scenes fetched at a time - keeps memory bounded on
     long date ranges or a server with limited RAM). A long date range is
     also searched a few years at a time and automatically retried if DEA's
     STAC gateway returns a transient error (occasionally a 502/503/504
     under load) - see "What if the run fails with a STAC/gateway error?"
     below.
   - **Run analysis.** A progress bar tracks each sensor's batches. This is
     the only step that talks to DEA - everything in tab 3 works instantly
     from what gets cached here, with no further fetching. The summary
     panel afterwards reports the actual measured size (in MB) of that
     in-memory cache - see "Is the live-filtering cache a memory problem on
     a server?" below.
3. **Tab 3 - Results.** Everything here is *live* - change a filter, click
   a checkbox, exclude a scene, and the raster/plots update immediately,
   no re-fetch:
   - **Data quality filters**: minimum valid-pixel %, maximum snow %,
     maximum cloud % (this one computed over the polygon itself, unlike
     tab 2's whole-tile version), and maximum smoke/haze % - all applied
     together to decide which fetched scenes count toward the raster and
     plots below. Every fetched scene still shows in the data table
     regardless.
   - **Pixel-mean raster**, draped over satellite imagery with the
     polygon's outline on top (dashed white), per sensor, with a GeoTIFF
     download button. NDVI's legend is fixed to its canonical -1 to 1
     range rather than autoscaling to the data, so colour is comparable
     across runs/polygons; the other metrics still autoscale to their own
     min/max since they don't have an equivalent fixed range.
   - **Time series**: mean ± SD error bars, one marker per scene, coloured
     by calendar month (● = Landsat, ◆ = Sentinel-2), with a month filter
     and independent linear-trend / GAM-smoothed-trend toggles (for
     tasselled cap wetness specifically, a dotted line at zero marks the
     rough threshold above which open water is typically present). Click,
     box-select or lasso-select points to select scenes, then either
     **add them to the scene viewer** (fetches each from DEA on demand,
     cropped to a square 200 m beyond the polygon on every side for
     landscape context rather than just the polygon's own tight bounding
     box; up to 5 scenes at a time, layered on one map - toggle individual
     scenes via the layer control or remove them below, adjust raster
     visibility/opacity, show true colour or the current metric, and
     download any of them as a GeoTIFF - the polygon is drawn as a dashed
     outline over the top, with an option to hide it) or **exclude/
     re-include** the selected scene(s) from the analysis. This viewer sits
     between the time-series and phenology plots.
   - **Snow cover over time**: a simple scatter of each fetched scene's
     snow cover over the polygon (no error bars - it's a single percentage,
     not an index) - click, box-select or lasso-select points here the same
     way as the time series, to add scenes to the viewer or exclude them.
     Unlike every other plot in tab 3, this one shows *every* fetched scene
     regardless of the quality filters above, since a heavily snow-covered
     scene has no valid pixels to plot an NDVI/tasselled-cap value for and
     would otherwise never appear anywhere to be inspected.
   - **Phenology plot**: every scene's value by month across all years,
     ordered July -> June (a water/growing year) rather than calendar
     Jan -> Dec, so the Southern Hemisphere's summer growing season stays
     together in one block instead of being split across the axis ends -
     with the same trend toggles (and the same TCW zero line) as the time
     series; the linear trend additionally prints its slope on the plot.
   - **Per-scene data table**, with an editable "excluded" checkbox column
     (an alternative to excluding via the chart) and a CSV download.
4. **Tab 4 - Trend comparisons.** A separate tool, independent of the
   polygon selected in tabs 1-3: upload two or more per-scene CSVs
   (exported from tab 3) and see them plotted as facets side by side -
   useful for comparing different polygons or different filter settings.
   Each file becomes its own facet, labelled with its filename. The same
   quality filters as tab 3 (month, minimum valid pixels, maximum snow/
   cloud/smoke %) are applied across every uploaded file before plotting -
   any column a given CSV doesn't have (an older export, or a hand-edited
   file) is simply skipped for that one check rather than excluding
   everything. Colour points by sensor, dataset, or month. Both plots have
   the same independent linear-trend / GAM-smoothed-trend toggles as tab
   3 - fit separately per dataset (per facet), since a single trend line
   spanning several different polygons or runs wouldn't mean anything. A
   **phenology plot** (mean value by month, July -> June, with the same
   per-dataset trend toggles and slope annotation) sits below the
   long-term trends plot, same as tab 3's.

## Why does 0% cloud cover return nothing?

DEA's `eo:cloud_cover` (tab 2's "maximum whole-tile cloud cover" slider) is
computed over the *entire satellite tile* - well over 100km across for
Landsat - not just your polygon. A tile can easily have some cloud on the
other side of that huge area while your peatland itself is perfectly clear, so
requiring 0% often matches nothing. That slider is really just a
fetch-volume limiter; the filter that actually reflects cloud over your
polygon is tab 3's "maximum scene cloud cover over polygon" slider, which
is computed from DEA's per-pixel `oa_fmask` clipped to the polygon itself.
Leave tab 2's slider reasonably generous (its default is 80%) and do your
real cloud filtering in tab 3.

## Why doesn't 100% minimum tile coverage exclude everything?

Tab 2's "Minimum tile coverage of polygon (%)" slider defaults to 100.
Coverage is computed as (intersection area of the scene's tile footprint
with your polygon) / (polygon area) - a floating-point calculation across
two reprojected geometries, so a scene that visibly, entirely covers the
polygon can come out as 0.999999... rather than a clean 1.0. To account for
this, a "100%" request is treated internally as "fully covers, allowing for
floating-point noise" (capped at 99.9%) rather than requiring
mathematically exact containment. Lower requests (50%, 90%, ...) aren't
affected by this cap.

## Why does the scene viewer's true-colour image sometimes look washed out?

If it does, it's worth reporting - the true-colour view is built by
percentile-stretching each band (2nd-98th percentile) over its own valid
pixels. Genuine no-data pixels are excluded from that stretch using DEA's
own `oa_fmask` no-data flag, which is more reliable than a single declared
nodata value across different products - this keeps out-of-coverage fill
(e.g. the edge of the 200m landscape-context buffer, past a scene's actual
swath) from skewing the colour stretch. Cloud, shadow and snow are
deliberately left visible rather than masked out.

## What if the run fails with a STAC/gateway error?

A run over a long date range makes several requests to DEA's STAC catalogue
to find out which scenes exist before fetching any of their pixels, and
DEA's STAC gateway occasionally returns a transient error under load (a
502/503/504, or a plain connection timeout) - not a problem with your
request, just a server-side blip. Two things make this less painful than
it sounds:

- **The search is already chunked and retried for you.** `run_analysis()`
  searches a few years at a time (`STAC_SEARCH_CHUNK_YEARS` in
  `modules/analysis.py`, default 3) rather than the whole date range in one
  request, and retries each chunk up to `STAC_SEARCH_MAX_ATTEMPTS` times
  (default 3, with a short backoff) if it hits exactly this kind of
  transient error. A single flaky request now costs a retry of a few
  years' worth of search, not the entire run.
- **A chunk that still fails after every retry doesn't take the rest of
  the run down with it.** It's skipped, and you get a specific warning
  under tab 2's results panel telling you exactly which period is missing
  scenes and why (rather than a bare stack trace and zero results) - the
  scenes from every other period you searched still come through. If you
  see one of these, the simplest fix is usually just running it again
  (a gateway timeout is transient by definition), or narrowing the date
  range if it keeps happening for the same period.

## Is the live-filtering cache a memory problem on a server?

Short answer: no, for the polygon sizes this app deals with. Only what
tab 3 needs raster data for (the pixel-mean raster) is kept in memory, and
the amount is small, bounded, and measured - it doesn't grow across a
session, either, only per run.

- **What's kept, and why.** Only the mean raster genuinely needs per-pixel
  data after the fetch is over - the time series, phenology plot, and every
  quality filter all work from `scene_df`'s per-scene summary numbers
  (mean/SD, snow/cloud/smoke %), which cost nothing extra to keep around.
  The smoke/haze detector briefly needs raw per-pixel data across multiple
  scenes at once (to build its clear-sky reference), but only for as long
  as `run_analysis()` is running - once `smoke_pct` is written into
  `scene_df`, those raw arrays are dropped immediately, before the function
  returns. `AnalysisResult.scene_cache` only ever holds each scene's four
  small metric arrays (NDVI/TCB/TCG/TCW).
- **How big, really.** Tab 2's summary panel reports the actual measured
  size of `AnalysisResult.scene_cache` (via `analysis.cache_size_bytes()`,
  real `numpy.nbytes`, not an estimate) for the run you just did. For a
  typical small peatland/fen polygon (tens to a few hundred pixels), even a
  full 40-year, thousand-scene run lands in the single-digit megabytes - a
  500-scene/400-pixel run measured around 3 MB with this cache design. It
  scales with polygon size x scene count, so a much larger polygon or a
  very long, high-frequency date range would use more - the reported
  figure is there so you can see the real number for your own runs rather
  than guessing.
- **Viewing scenes costs nothing extra to the main cache.** Adding a scene
  to the viewer doesn't touch `scene_cache` at all - `fetch_single_scene()`
  re-fetches just that one scene from DEA on demand, and the app holds at
  most `MAX_VIEWED_SCENES` (5) of them at a time; adding a 6th is rejected
  with a warning until you remove one. Download anything you want to keep
  (a GeoTIFF button sits under the map for each scene) before removing it
  or starting a new analysis run, which clears the whole list.
- **Per-session, not global.** Each user's cache lives in their own
  `st.session_state` and isn't shared. It gets replaced (the old result
  becomes garbage-collectable) as soon as that user runs a new polygon or
  reloads - it doesn't accumulate indefinitely within one session.
- **What matters for server hosting.** On a shared/multi-user deployment,
  total memory scales with concurrent active sessions x each one's cache
  size, not with any single run in isolation.

## Deploying to Streamlit Community Cloud

1. Push this `Peatland_trends/` folder to a GitHub repo - it's
   self-contained (vendored DEA-access code, the two shapefiles, product
   config, everything under `modules/`) other than the Python packages in
   `requirements.txt`. `.gitignore` already excludes `.venv/` and
   `__pycache__/`, which shouldn't be committed.
2. On [share.streamlit.io](https://share.streamlit.io), click "New app",
   point it at that repo, and set the main file path to `app.py`. If
   `Peatland_trends/` isn't the repo root, set "Main file path" to
   `Peatland_trends/app.py` (and the working directory the same way, in
   "Advanced settings") - or make this folder its own repo.
3. No secrets or API keys are needed. DEA's STAC endpoint
   (`https://explorer.sandbox.dea.ga.gov.au/stac`) is public and anonymous
   - there's nothing to add under the app's "Secrets" settings.
4. `runtime.txt` pins Python 3.11, which has readily-available prebuilt
   wheels for the geospatial dependencies (rasterio, GDAL-backed
   geopandas, etc.).
5. `requirements.txt` uses `opencv-python-headless`, not `opencv-python` -
   the plain package pulls in a GUI dependency (`libGL.so.1`) that
   Streamlit Cloud's container doesn't have, which breaks the build the
   moment anything imports `cv2` (here, indirectly, via
   `vendor/rsderiv/sen1.py`). If a build ever fails on a GDAL/rasterio/
   geopandas wheel instead, a `packages.txt` with the relevant apt-level
   GDAL packages may be needed, though modern wheels usually bundle
   everything they need.
6. `requirements.txt` uses `>=` lower bounds rather than exact pins, so a
   future rebuild could pick up newer package versions than this app was
   last tested against. That's the first thing to check if a
   previously-working deployment stops building after a period with no
   changes to this repo.

## Folder layout

- `app.py` - the Streamlit UI (a Home tab plus 4 numbered tabs: polygon
  picker, fetch controls, live-filterable results, CSV trend comparisons).
- `modules/polygons.py` - loads and merges the two shapefiles into one
  GeoDataFrame with a common `uid`/`source`/`label`/`area_ha` schema.
- `modules/indices.py` - NDVI and Tasselled Cap Brightness/Greenness/Wetness
  calculations (Crist 1985 coefficients, applied to both sensors for a
  directly comparable cross-sensor time series).
- `modules/analysis.py` - the DEA fetch + per-scene caching pipeline, plus
  the live re-filtering functions (`filter_scene_df`,
  `recompute_pixel_rasters`), the single-scene re-fetch used by the scene
  viewer (`fetch_single_scene`), `cache_size_bytes` (measures the actual
  in-memory footprint of the scene cache, surfaced in tab 2), and
  `filter_uploaded_df` (tab 4's equivalent of `filter_scene_df`, but
  forgiving of uploaded CSVs missing a column). See its module docstring
  for the two-stage fetch/filter design.
- `modules/smoke_detection.py` - per-pixel temporal-anomaly (smoke/haze)
  detection.
- `modules/trends.py` - shared linear-regression (with slope) and GAM
  (via `pygam`) trend-line helpers used by both the time-series and
  phenology plots. `gam_trend()` returns an explicit error string when a
  fit genuinely fails, rather than a generic "not enough points" message
  that could be misleading if the real cause is something else.
- `modules/raster_viz.py` - reprojects rasters (single-band metric or
  3-band true-colour) to WGS84 and turns them into colour-mapped PNG
  overlays + legends for display on the satellite maps in tab 3.
- `vendor/` - self-contained copies of the `STACDataManager` /
  `RSDataProductManager` DEA-access classes (see `vendor/README.md`).
- `config/products.json` - Landsat and Sentinel-2 product definitions,
  extended with a `canonical_bands` map per product so `analysis.py` can
  compute indices from each sensor's differently-named bands.
- `alpinebogs_wetheathlands_AllVic_gda94.*`,
  `Biodiversity_BogsAn_Fens_Snowy_Mountains.*` - the two polygon layers,
  bundled with the app since they're what tab 1's map picker loads from
  (`modules/polygons.py`) - no external data source needed at runtime
  beyond DEA itself.
- `runtime.txt`, `.gitignore` - deployment config, see "Deploying to
  Streamlit Community Cloud" above.

## Design notes

**Fetch once, filter forever (live) - but only cache what's still needed.**
`analysis.run_analysis()` (tab 2) fetches from DEA and writes a per-scene
summary row (means, snow/cloud/smoke %) into `scene_df` for every scene -
it does not bake in any valid-pixel/snow/cloud/smoke threshold. Tab 3's
quality filters (`filter_scene_df()`) work entirely from that dataframe, so
they, the time series, and the phenology plot never touch raster data at
all. The one thing that still needs per-pixel data afterwards is the
pixel-mean raster, so `recompute_pixel_rasters()` reads each scene's small
metric arrays (NDVI/TCB/TCG/TCW) from `AnalysisResult.scene_cache`. The DEA
*request* itself is also batched (`batch_size` scenes fetched and
processed at a time) so memory stays bounded regardless of date-range
length.

**Per-pixel cloud/snow masking, not whole-scene rejection.** Rather than
dropping an entire scene if any pixel over the polygon is cloud-affected,
this app masks cloud/shadow/snow/no-data pixels individually (DEA's
`oa_fmask`, keeping only classes 1/5 = clear/water as usable for the index
maths) and computes each scene's mean/SD from whatever pixels remain. Snow
and cloud percentages over the polygon (`snow_pct`, `cloud_pct`) are
tracked per scene regardless, and tab 3's sliders let you reject whole
scenes above a chosen threshold for either.

**Smoke/haze detection.** DEA's `oa_fmask` doesn't flag smoke or thin haze
- both can still elevate NDVI-relevant reflectance without ever being
classified as cloud. `modules/smoke_detection.py` builds a per-pixel
"clear-sky reference" (the low percentile of blue-band reflectance seen at
that exact pixel across every other scene fmask already called clear/water)
and flags a scene where a meaningful fraction of pixels sit well above
their own reference. This is a starting-point heuristic, not validated for
these specific sites - use tab 3's single-scene true-colour viewer to
sanity-check any scene the smoke filter flags before trusting it, and
adjust tab 2's sensitivity threshold accordingly.

**All-touched pixel rasterisation.** Many of these peatland/fen polygons are
tiny - the smallest Snowy Mountains fens are ~0.007 ha, well under a single
Landsat pixel (0.09 ha). The polygon-to-grid mask is rasterised with
`all_touched=True` (any pixel the polygon boundary touches counts), not
the more common "pixel centre inside polygon" rule, which would silently
return zero pixels for a lot of these polygons.

**Full tile-coverage filtering at fetch time.** A scene can have low
whole-tile cloud cover and still contribute nothing over a small polygon
if the polygon sits right at the edge of that scene's swath. `analysis.py`
filters scenes by tile-polygon overlap before fetching any raster data,
default 100% (tile must fully cover the polygon).

**Tasselled Cap coefficients applied identically to Landsat and
Sentinel-2**, for a directly comparable cross-sensor time series rather
than sensor-specific absolute accuracy - see `modules/indices.py`.

**Click-to-view and manual exclusion both write to the same state.**
Selecting points on the time-series chart and clicking "Exclude", or
ticking the "excluded" checkbox in the per-scene table, both update the
same underlying set - whichever you use, the raster/plots update the same
way. Excluding a scene doesn't delete it from `scene_df`/the CSV export;
it's just left out of the aggregated raster/time-series/phenology views
(and clearly marked in the table).

**Trend comparisons (tab 4) work from plain CSVs, not the live app
state.** It expects the same columns tab 3's CSV export produces
(`datetime`, `sensor`, `<metric>_mean`, `<metric>_std`) but doesn't care
where the CSV came from - you could compare runs from different days,
different people, or hand-edited files, as long as the columns match.

**Tab 4's trend lines are fit per dataset, not across all of them at
once.** Both faceted plots are built with `plotly.subplots.make_subplots`
(one subplot per uploaded dataset) so each facet's linear/GAM trend is
computed from that dataset's own points only. If a particular dataset
doesn't have enough points to fit a trend, that's reported per-dataset
underneath the plot rather than silently skipped or blended into an
average across every other dataset.

**Map performance.** The two layers combined are ~12,000 polygons. The
map uses simplified display geometry (the original, unsimplified geometry
is always used for the actual analysis clip) and layer toggles to help,
but the sidebar's searchable polygon selector is the more reliable way to
pick a specific polygon.

**Maps render correctly across every tab.** Streamlit's `st.tabs()` mounts
every tab's content at once and hides inactive ones with CSS - switching
tabs never re-runs the underlying script, and a Folium map that renders
into a still-hidden tab would otherwise mount into a zero-pixel container
and end up zoomed out to a full, sideways-tiled world view. Every map in
this app (`fix_map_visibility_zoom()` in `app.py`) uses a `ResizeObserver`
on the map's own container that fires the moment it actually becomes
visible and re-runs `invalidateSize()` + `fitBounds()` at that point.
