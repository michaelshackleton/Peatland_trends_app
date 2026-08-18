from collections import defaultdict
from shapely.geometry import box, shape
from shapely.ops import unary_union
import numpy as np,geopandas as gpd,rioxarray as rxr, xarray as xr
from rasterio.features import rasterize

def filter_items_bbox_tolerance(item_list,bbox_array,tolerance=0.998):
    bbox = box(*bbox_array)
    # Group items by date
    items_by_date = defaultdict(list)
    for item in item_list:
        # Extract date from item.datetime (already a datetime object)
        item_date = item.datetime.date()
        items_by_date[item_date].append(item)

    # Filter dates where combined geometry fully contains the bbox
    valid_dates = []
    for date, items in items_by_date.items():
        # Combine geometries of all items for this date (in EPSG:4326)
        item_geometries = [shape(item.geometry) for item in items]
        combined_geometry = unary_union(item_geometries)
        # Check if combined geometry contains the bbox (in EPSG:4326)
        overlap_prop = combined_geometry.intersection(bbox).area/bbox.area
        if overlap_prop >= tolerance:
            valid_dates.append(date)
            # print(f"Date {date}: Combined geometry contains the bbox ({len(items)} items)")
        # else:
            # print(f"Date {date}: Combined geometry does NOT contain the bbox ({len(items)} items) overlap %:" + str(overlap_prop))

    # Optionally, collect items for valid dates
    filtered_items = [
        item
        for date in valid_dates
        for item in items_by_date[date]
    ]
    return filtered_items
def convert_vector_to_raster(gdf,ref_rxr_raster,value_column=None, fill_value=0, dtype=np.int8, name='rasterized_data'):
    ref_crs = ref_rxr_raster.rio.crs
    if gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)
    # Get reference raster properties
    ref_shape = ref_rxr_raster.shape
    if len(ref_shape)==3:              # Handle time dimension in reference raster
        ref_shape=ref_shape[1:]
    ref_transform = ref_rxr_raster.rio.transform()
    ref_coords = {
        'x': ref_rxr_raster.coords['x'],
        'y': ref_rxr_raster.coords['y'],
        'spatial_ref': ref_rxr_raster.coords['spatial_ref']
    }
    if value_column is not None:
        if value_column not in gdf.columns:
            raise ValueError(f"Column '{value_column}' not found in GeoDataFrame")
        shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[value_column])]
    else:
        shapes = [(geom, 1) for geom in gdf.geometry]
    rasterized = rasterize(
        shapes=shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=fill_value,
        dtype=dtype
    )

    # Convert to xarray.DataArray and assign coordinates
    raster_da = xr.DataArray(
        data=rasterized,
        coords=ref_coords,
        dims=('y', 'x'),
        name=name
    )
    # Set CRS
    raster_da.rio.write_crs(ref_crs)

    # Verify alignment
    if not np.array_equal(raster_da.coords['x'], ref_coords['x']) or \
       not np.array_equal(raster_da.coords['y'], ref_coords['y']) or \
       raster_da.shape != ref_shape or \
       raster_da.rio.crs != ref_crs:
        raise ValueError("Rasterized output does not align with reference raster")

    return raster_da
