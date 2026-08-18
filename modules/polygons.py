"""
Loads the two peatland spatial layers this app selects analysis polygons
from, and merges them into a single GeoDataFrame with a common schema
(uid / source / label / area_ha / geometry) so the rest of the app doesn't
need to know about each shapefile's own idiosyncratic columns.

Layers:
- alpinebogs_wetheathlands_AllVic_gda94.shp - Victorian alpine bogs & wet
  heathlands (source uid prefix "AB").
- Biodiversity_BogsAn_Fens_Snowy_Mountains.shp - Snowy Mountains (NSW) bogs
  & fens (source uid prefix "SM").

Both are reprojected to WGS84 (EPSG:4326) for display on a Leaflet map.
Geometry stored in the returned GeoDataFrame is full precision (not
simplified) - it's what's used both for the map and for clipping rasters
during analysis, so simplifying it would silently shrink/distort the actual
analysis area. A separately-simplified copy for fast map rendering is
provided via `display_geometry()`.
"""

from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd

WGS84 = "EPSG:4326"

ALPINE_BOGS_SHP = "alpinebogs_wetheathlands_AllVic_gda94.shp"
SNOWY_MTNS_SHP = "Biodiversity_BogsAn_Fens_Snowy_Mountains.shp"

SOURCE_ALPINE_BOGS = "Alpine Bogs & Wet Heathlands (Vic)"
SOURCE_SNOWY_MTNS = "Bogs & Fens (Snowy Mountains, NSW)"

# Simplification tolerance (degrees, ~ metres at these latitudes) used only
# for the map display copy of the geometry - NOT used for analysis.
DISPLAY_SIMPLIFY_TOLERANCE = 0.00005


def _load_alpine_bogs(base_dir: str) -> gpd.GeoDataFrame:
    path = os.path.join(base_dir, ALPINE_BOGS_SHP)
    gdf = gpd.read_file(path).to_crs(WGS84)
    out = gpd.GeoDataFrame(
        {
            "uid": "AB-" + pd.RangeIndex(len(gdf)).astype(str),
            "source": SOURCE_ALPINE_BOGS,
            "label": "Peatland " + gdf["BogID"].astype("Int64").astype(str)
            + " (" + gdf["MAPPING_UN"].fillna("unmapped").astype(str) + ")",
            "area_ha": pd.to_numeric(gdf["HECTARES"], errors="coerce"),
        },
        geometry=gdf.geometry,
        crs=WGS84,
    )
    return out


def _load_snowy_mountains(base_dir: str) -> gpd.GeoDataFrame:
    path = os.path.join(base_dir, SNOWY_MTNS_SHP)
    gdf = gpd.read_file(path).to_crs(WGS84)
    # OBJECTID isn't reliably unique in this shapefile (a handful of rows
    # share OBJECTID 0), so the uid is built from the row's own position in
    # the file instead - guaranteed unique regardless of attribute quality.
    out = gpd.GeoDataFrame(
        {
            "uid": "SM-" + pd.RangeIndex(len(gdf)).astype(str),
            "source": SOURCE_SNOWY_MTNS,
            "label": gdf["Swamp_name"].fillna("").astype(str).str.strip()
            + " – " + gdf["Bog_name"].fillna("").astype(str).str.strip(),
            "area_ha": pd.to_numeric(gdf["Hectares"], errors="coerce"),
        },
        geometry=gdf.geometry,
        crs=WGS84,
    )
    return out


def load_polygons(base_dir: str) -> gpd.GeoDataFrame:
    """Loads and merges both peatland layers. `base_dir` is the folder
    containing the two .shp files (this app's own folder)."""
    alpine = _load_alpine_bogs(base_dir)
    snowy = _load_snowy_mountains(base_dir)
    merged = pd.concat([alpine, snowy], ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=WGS84)
    # Drop empty/invalid geometries defensively - a handful of shapefiles
    # like these sometimes carry a stray null/empty geometry row.
    merged = merged[~merged.geometry.is_empty & merged.geometry.notna()].reset_index(drop=True)
    return merged


def display_geometry(gdf: gpd.GeoDataFrame, tolerance: float = DISPLAY_SIMPLIFY_TOLERANCE):
    """Returns a simplified copy of `gdf`'s geometry column for fast map
    rendering. Analysis code should always use the original `gdf.geometry`
    (looked up by uid), never this simplified version."""
    return gdf.geometry.simplify(tolerance, preserve_topology=True)


def get_geometry_by_uid(gdf: gpd.GeoDataFrame, uid: str):
    """Returns the full-precision (lon/lat, EPSG:4326) Shapely geometry for
    a single polygon by its uid."""
    row = gdf.loc[gdf["uid"] == uid]
    if len(row) == 0:
        raise ValueError(f"No polygon found with uid={uid!r}")
    return row.geometry.iloc[0]
