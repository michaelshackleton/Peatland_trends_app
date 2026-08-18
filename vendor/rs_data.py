import numpy as np
import os, json, pandas as pd, geopandas as gpd
import rioxarray as rxr
import xarray as xr
from shapely.geometry import shape
import pystac_client
import planetary_computer
import utils
import odc.stac
from odc.stac import configure_rio
from datetime import datetime

class STACDataManager:
    def __init__(self,root_dir):
        self.root_dir = root_dir
        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)
    def make_dir_path(self,collection):
        pass
    def make_file_name(self,product,collection,date,region_code,sub_region_code = None):
        """Make file path using the following naming convention:
            Directory - <root directory>/<source collection>/<region code>/<date>
            File name - <Source collection>_<Processed product>_<Region code>-<Sub-region code>_<Date>.<Output extension>
            NOTE:
                Underscores are stripped in the file name to enable splitting on them when reading file names
        """
        date_format = np.datetime_as_string(date,unit='D')
        dir = os.path.join(self.root_dir,collection,region_code,date_format)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if sub_region_code is None:
            reg = region_code.replace("_",'')
        else:
            reg = region_code.replace("_",'') + "-" +str(sub_region_code).replace("_",'')
        file_name = collection.replace("_",'') + "_" + str(product).replace("_",'') + "_" + reg + "_" + date_format + ".tif"
        return(os.path.join(dir,file_name))
    def stac_ds_to_product(self,ds,product_manager,region_manager,sub_region_code = None, process_only=True,compress=True,split_date=True):
        """Writes xarray dataset to a tiff for each time coord"""
        ds = product_manager.make_product_dask(ds,region_manager,sub_region_code)
        if process_only:
            return ds
        else:
            compress_type = 'lzw' if compress else None
            if not split_date:
                ds = ds.compute()
            time_values = np.atleast_1d(ds.time.values)
            if 'time' in ds.coords and 'time' in ds.dims:
                for t in time_values:
                    d = ds.sel(time=t)
                    collection = str(d.collection_id.values)
                    fname = self.make_file_name(product_manager.product_code,collection,t,region_manager.region_code,sub_region_code)
                    try:
                        d.rio.to_raster(fname,compress=compress_type)
                    except Exception as e:
                        print(f"Error: {e}")
            else:
                collection = str(ds.collection_id.values)
                fname = self.make_file_name(product_manager.product_code,collection,time_values[0],region_manager.region_code,sub_region_code)
                try:
                    ds.rio.to_raster(fname,compress=compress_type)
                except Exception as e:
                    print(f"Error: {e}")
    def write_deriv_product(self,ds,product_manager,region_manager,sub_region_code=None,clip_region=True,split_date=True, compress=True):
        compress_type = 'lzw' if compress else None
        ds.rio.write_nodata(product_manager.nodata,inplace=True)
        if clip_region:
            geom = region_manager.get_polygon(sub_region_code,target_crs = ds.rio.crs.to_epsg())
            ds = ds.rio.clip([geom])
        if not split_date:
            ds.compute()
        time_values = np.atleast_1d(ds.time.values)
        if 'time' in ds.coords and 'time' in ds.dims:
            for t in time_values:
                d = ds.sel(time=t)
                fname = self.make_file_name(product_manager.product_code,'deriv',t,region_manager.region_code,sub_region_code)
                try:
                    d.rio.to_raster(fname,compress=compress_type)
                except Exception as e:
                    print(f"Error: {e}")
        else:
            fname = self.make_file_name(product_manager.product_code,'deriv',time_values[0],region_manager.region_code,sub_region_code)
            try:
                ds.rio.to_raster(fname,compress=compress_type)
            except Exception as e:
                print(f"Error: {e}")
    # def write_xarray(xrdata,out_file,compress = True):
    #     band_names = xrdata.band.values.tolist()  # Convert to list
    #     xrdata.attrs['long_name'] = [str(band) for band in band_names]  # Ensure strings
    #     compress_type = 'lzw' if compress else None
    #     xrdata.rio.to_raster(out_file,driver='GTiff',compress=compress_type)

    def get_stac_items(self,start_date,end_date,product_manager,region_manager,sub_region=None,tolerance=0.998):
        """
        Given an analysis region and required data product, return the available online data within a given date range
        """
        extent = region_manager.get_region_extent(sub_region=sub_region,target_crs=4283)
        if product_manager.source=='DEA':
            url = 'https://explorer.sandbox.dea.ga.gov.au/stac'
            catalog = pystac_client.Client.open(url)
        elif product_manager.source=='MPC':
            url = 'https://planetarycomputer.microsoft.com/api/stac/v1'
            catalog = pystac_client.Client.open(url,modifier=planetary_computer.sign_inplace)
        else:
            raise ValueError("Source parameter is not valid. Must be DEA for collections from Digital Earth Australia or MPC for collections from Microsoft Planetary Computer")
        search = catalog.search(
            collections=product_manager.collections,
            datetime=f"{start_date}/{end_date}",
            bbox = extent
        )
        items = list(search.items())
        return(items)
    def stac_items_to_gdf(self,items,region_manager = None,target_crs = None,sub_region=None):
        """Given a list of STAC items from get_stac_items(), return a geopandas dataframe with STAC properties and data extent polygons"""
        i = []
        for item in items:
            d = item.properties
            d['geometry'] = item.geometry
            d['collection'] = item.collection_id
            i.append(d)
        df = pd.DataFrame(i)
        df['geometry'] = df['geometry'].apply(lambda x: shape(x) if x else None)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        gdf['datetime'] = pd.to_datetime(gdf['datetime'], format='mixed', utc=True)
        if all([region_manager,target_crs]):
            gdf = gdf.to_crs(target_crs)
            geom = region_manager.get_polygon(sub_region=sub_region,target_crs=target_crs)
            gdf['region_overlap'] = gdf.geometry.intersection(geom).area/geom.area
        return gdf
    def get_local_available_data(self,start_date,end_date,region_manager,product_manager=None):
        """
        Function to search the local directory structure and return a dataframe of available images
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        if product_manager is not None:
            collections = product_manager.collections
        data_list = []
        for collection in collections:
            collection_path = os.path.join(self.root_dir, collection)
            if not os.path.exists(collection_path):
                # print(f"Collection directory {collection_path} does not exist.")
                continue
            region_path = os.path.join(collection_path, region_manager.region_name)
            if not os.path.exists(region_path):
                continue
            for dir in os.listdir(region_path):
                fol_date = pd.to_datetime(dir)
                if start_date <= fol_date <= end_date:
                    date_path = os.path.join(region_path,dir)
                    files = [f for f in os.listdir(date_path) if f.endswith('.tif')]
                    for file in files:
                        file_prod = file.split('_')[1]
                        if file_prod==product_manager.product_code:
                            full_path = os.path.join(date_path,file)
                            data_list.append({'date':pd.to_datetime(fol_date),'collection':collection,'product':product_manager.product_code,'dir_path':date_path,'file_path':full_path,'file_name':file})
        return(pd.DataFrame(data_list))
    def stac_items_to_xrdataset(self,items,product_manager,region_manager,sub_region=None,target_crs=32755,group=None):
        """

        """
        configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
        extent = region_manager.get_region_extent(sub_region=sub_region,target_crs=4283)
        # Load the stac items
        data = odc.stac.stac_load(
                    items,
                    bands=product_manager.bands_to_include,
                    bbox=extent,crs='EPSG:'+str(target_crs),
                    resolution=product_manager.resolution,
                    resampling=product_manager.bands_to_resample,chunks={},
                    groupby=group
                    )
        collection_dict = {}
        # Process the items to assign the collection ID as coord of the dataset.
        # This is important for writing and naming the file correctly if saved locally
        for item in items:
            dt = datetime.fromisoformat(item.properties['datetime'].replace('Z', '+00:00'))
            formatted_dt = dt.strftime('%Y-%m-%dT%H:%M:%S')
            collection = item.collection_id
            collection_dict[formatted_dt] = collection
        date_strings = [np.datetime_as_string(t, unit='s') for t in data.time.values]
        collection_ids = [collection_dict.get(date, '') for date in date_strings]
        data = data.assign_coords({'collection_id':('time', collection_ids)})
        return(data)

    def local_data_to_xarray(self,df):
        """
        Given a dataframe of available images, convert them to an xarray dataset, indexed by time
        """
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        # List to store individual xarray Datasets
        datasets = []
        # Load each file and assign date as time coordinate
        if isinstance(df,pd.Series):
            ds = rxr.open_rasterio(df['file_path'], masked=False,chunks=True)
            ds = ds.assign_coords(band=list(ds.attrs['long_name'])).to_dataset('band')
            # If the dataset has a 'band' dimension, convert it to variables or select data as needed
            if 'band' in ds.dims:
                ds = ds.squeeze('band', drop=True)  # Example: Remove band dimension if single-band
            ds = ds.expand_dims(time=[df['date']])
            return(ds)
        elif isinstance(df,pd.DataFrame):
            for _, row in df.iterrows():
                ds = rxr.open_rasterio(row['file_path'], masked=False,chunks=True)
                ds = ds.assign_coords(band=list(ds.attrs['long_name'])).to_dataset('band')
                # If the dataset has a 'band' dimension, convert it to variables or select data as needed
                if 'band' in ds.dims:
                    ds = ds.squeeze('band', drop=True)  # Example: Remove band dimension if single-band
                ds = ds.expand_dims(time=[row['date']])
                datasets.append(ds)
                # Combine datasets along time dimension
            combined_ds = xr.concat(datasets, dim='time')
            return combined_ds
        else:
            raise ValueError("Must pass either a pandas dataframe or series with 'file_path' and 'date' columns")
    def filter_stac_items_tiles(self,items,tiles):
        """Filter the items returned by get_stac_items() based on their ODC region code (i.e. tile code)"""
        if type(tiles)==list:
            tiles = np.array(tiles)
        elif type(tiles)==str:
            tiles = np.array([tiles])
        elif type(tiles)==np.ndarray:
            pass
        else:
            raise ValueError("Must pass string, list or numpy array containing required tiles")
        filtered_items = []
        for item in items:
            if np.isin(item.properties['odc:region_code'],tiles):
                filtered_items.append(item)
        return filtered_items
    def filter_stac_items_region_overlap(self,items,min_overlap_pct,region_manager,target_crs,sub_region=None):
        item_gdf = self.stac_items_to_gdf(items)
        if 'region_overlap' not in item_gdf.columns:
            item_gdf = item_gdf.to_crs(target_crs)
            geom = region_manager.get_polygon(sub_region=sub_region,target_crs=target_crs)
            overlap = item_gdf.geometry.intersection(geom).area/geom.area
        else:
            overlap = item_gdf['overlap']
        filtered_items = []
        for i in range(0,len(overlap)):
            if overlap[i]>=min_overlap_pct:
                filtered_items.append(items[i])
        return filtered_items


    def filter_stac_items_eocloud(self,items,max_cloud):
        """
            Filter items by the total cloud cover over the whole tile.
            NOTE: Use filter_stac_items_fmask_qa() to filter by cloud cover over the analysis region
        """
        filtered_items = []
        for item in items:
            if item.properties['eo:cloud_cover']<=max_cloud:
                filtered_items.append(item)
        return filtered_items

    def filter_stac_items_fmask_qa(self,items,region_manager,product_manager,min_overlap,min_valid):
        """
            Filter the items returned by get_stack_items() by the coverage of valid pixels using the F-Mask QA data.
            Uses qa_daily_fmask_eval() to get the proportion of clear pixels and proportion of overlap of the analysis region.
            Users can then specify the minimum overlap and minimum % of clear pixels to filter items in the item list.
            Note:
                Minimum overlap: refers to the proportion that data from a single date overlap the analysis region. This means
                                 that all tiles for a single date are merged together. Where an analysis region spans tiles collected on different
                                 days, the percetage will be less than 100.
                Minimum valid:   refers to the proportion of pixels for data overlapping the study area on single date that are clear.
                                 Where an analysis region spans tiles collected on different days, this percentage is relative to the available data
                                 for that date. It is not relative to the whole study area unless the data overlap = 100%.
        """
        qa_dates = self.qa_daily_fmask_eval(items,region_manager,product_manager).compute()
        qa_dates = qa_dates.where(qa_dates.date[(qa_dates['extent_coverage_pct']>=min_overlap )& (qa_dates['valid_pixel_pct']>=min_valid)])
        filtered_items = []
        for item in items:
            if np.isin(item.datetime.date(),qa_dates.date):
                filtered_items.append(item)
        return filtered_items
    def qa_daily_fmask_eval(self,items,region_manager, product_manager):
        """
            Cal
        """

        oa_ds = self.stac_items_to_xrdataset(items,product_manager=product_manager,region_manager=region_manager)
        # Non-nan mask per tile/date
        available_pixels_td = (~oa_ds['oa_fmask'].isnull()).astype(np.int8)
        if region_manager is not None:
            # Restrict available pixels to the regions analysis area
            gdf = region_manager.region_poly
            analysis_area_mask = utils.convert_vector_to_raster(gdf,oa_ds['oa_fmask'])
            analysis_area = analysis_area_mask.sum()
            available_pixels_td = ((available_pixels_td==1) & (analysis_area_mask==1)).astype(np.int8)
        else:
            # Analysis area just whole extent if no polygon specified
            analysis_area = oa_ds['oa_fmask'].shape[1]*oa_ds['oa_fmask'].shape[2]
        # Create valid pixel mask (non-cloud) per tile/date within analysis area
        valid_mask_td = (oa_ds['oa_fmask'].isin([1,4,5])).astype(np.int8)
        # Ensure pixels not available for analysis set to zero (e.g. outside region or nan)
        valid_mask_td = valid_mask_td.where(available_pixels_td==1,0)
        # Merge to get daily valid & available masks. Using max() ensures 1 is selected for each
        valid_mask_date = valid_mask_td.groupby("time.date").max()
        available_pixels_date = available_pixels_td.groupby("time.date").max()
        # Calculate the percent of the analysis area the daily mask covers
        extent_coverage_pct = available_pixels_date.sum(dim=['x','y'])/analysis_area
        # Calculate the percent of pixels that are valid within each date. Expressed as a % relative to
        # the available pixels for that date, not the total analysis area.
        valid_pixel_pct_date = valid_mask_date.sum(dim=['x','y'])/available_pixels_date.sum(dim=['x','y'])
        results_timeseries =  xr.Dataset({
                        'extent_coverage_pct': extent_coverage_pct,
                        'valid_pixel_pct': valid_pixel_pct_date
                    })
        results_timeseries['extent_coverage_pct'].attrs = {
            'description': 'Fraction of analysis area covered by available (non-null) pixels per date',
            'units': 'fraction'
        }
        results_timeseries['valid_pixel_pct'].attrs = {
            'description': 'Fraction of available pixels that are valid (non-cloud, non-null) per date',
            'units': 'fraction'
        }
        return results_timeseries
def read_xarray(file_path):
    xrdata = rxr.open_rasterio(file_path)
    xrdata = xrdata.assign_coords(band=list(xrdata.attrs['long_name']))
    return(xrdata)
class STACRSRegionManager:
    def __init__(self,region_name,region_param_path = None,region_code=None,region_poly_path=None,layer=None):
        self.root_dir = None
        self.region_poly = None
        self.sub_region_col = None
        self.region_name = region_name
        self.region_code = None
        if region_param_path is not None:
            if os.path.exists(region_param_path):
                with open(region_param_path) as f:
                    region_params = json.load(f)
                if region_name in region_params:
                    self.region_code = region_params[region_name]['code']
                    poly_layer = None
                    if 'reg_shape_path' in region_params[region_name]:
                        if 'reg_shape_layer' in region_params[region_name]:
                            poly_layer = region_params[region_name]['reg_shape_layer']
                        self.load_shape(region_params[region_name]['reg_shape_path'],layer=poly_layer)
                    else:
                        raise ValueError("Path to region extent dataset is not available in region parameter file")
                else:
                    raise ValueError("Region name is not in the region parameter file")
            else:
                raise ValueError("Region parameter file does not exist")
        elif any([region_code,region_poly_path]) is None:
            raise ValueError("Must pass STACDataManager object or ")
        else:
            self.region_code = self.region_code
            self.load_shape(region_poly_path,layer)
    def load_shape(self,shape_path,layer=None):
        """Load in the polygons used to define the spatial analysis region"""
        if os.path.exists(shape_path):
            if shape_path.endswith(".shp"):
                self.region_poly = gpd.read_file(shape_path)
            elif shape_path.endswith(".gpkg"):
                if layer is None:
                    raise ValueError("Region extent dataset is a geopackage but the layer name was not specified")
                else:
                    self.region_poly = gpd.read_file(shape_path,layer=layer)
        else:
            raise ValueError("File for the region extent dataset does not exist")
    def get_polygon(self,sub_region=None,target_crs=None):
        """Return the geometry of the analysis region or sub-region if specified"""
        if sub_region is None:
            if target_crs is None:
                return self.region_poly.iloc[0].geometry
            else:
                return self.region_poly.to_crs(target_crs).iloc[0].geometry
        else:
            sub_reg_geom = self.region_poly.loc[self.region_poly[self.sub_region_col]==sub_region]
            if len(sub_reg_geom)==0:
                raise ValueError("Specified sub-region does not exist in the region extent dataset")
            else:
                if target_crs is None:
                    return self.region_poly.loc[self.region_poly[self.sub_region_col]==sub_region].geometry
                else:
                    return self.region_poly.loc[self.region_poly[self.sub_region_col]==sub_region].to_crs(target_crs).geometry
    def get_region_extent(self,sub_region=None,target_crs=None):
        """Return the total bounds of the analysis region or sub-region if specified"""
        if sub_region is None:
            if target_crs is None:
                return self.region_poly.total_bounds
            else:
                return self.region_poly.to_crs(target_crs).total_bounds
        else:
            sub_reg_geom = self.region_poly.loc[self.region_poly[self.sub_region_col]==sub_region]
            if len(sub_reg_geom)==0:
                raise ValueError("Specified sub-region does not exist in the region extent dataset")
            else:
                if target_crs is None:
                    return self.region_poly.loc[self.region_poly[self.sub_region_col]==sub_region].total_bounds
                else:
                    return self.region_poly.loc[self.region_poly[self.sub_region_col]==sub_region].to_crs(target_crs).total_bounds
    def make_nodata_mask(self,ref_xrarray,sub_region=None):
        gdf = self.region_poly
        if sub_region is not None:
            gdf = gdf.loc[gdf[self.sub_region_col]==sub_region]
        mask = utils.convert_vector_to_raster(gdf,ref_xrarray)
        return mask

if __name__ == '__main__':
    pass
