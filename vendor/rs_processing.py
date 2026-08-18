import os

from rioxarray.merge import merge_arrays
from rsderiv import sharpen
import os
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from rsderiv import sen1
import json

#TODO: Abstract product manager required
class RSDerivativeProductManager:
    def __init__(self,product_code,nodata,dtype):
        self.product_code = product_code
        self.nodata = nodata
        self.dtype = dtype
        self.process = "nil"

class RSDataProductManager:
    def __init__(self,product_code,product_param_path=None,source=None,collections=None,resolution=None,
                 bands_to_include=None,bands_to_resample=None,bands_to_sharpen=None,nodata=None,process=None,dtype=None):
        self.product_code = product_code
        if product_param_path is not None:
            if os.path.exists(product_param_path):
                with open(product_param_path) as f:
                    product_params = json.load(f)[product_code]
                    self._validate_inputs(product_params['source'],product_params['collections'],product_params['resolution'],product_params['bands_to_include'],
                                          product_params['bands_to_resample'],product_params['bands_to_sharpen'],product_params['nodata'],product_params['process'],product_params['dtype'])
            else:
                raise ValueError("Specified product parameter file does not exist")
        elif any(product_code,source,collections,resolution,bands_to_include,bands_to_resample,bands_to_sharpen,nodata,process) is None:
            raise ValueError("Did not specify enough parameters")
        else:
            self.product_code = product_code
            self._validate_inputs(source,collections,resolution,bands_to_include,bands_to_resample,bands_to_sharpen,nodata,process,dtype)
    def _validate_inputs(self,source,collections,resolution,bands_to_include,bands_to_resample,bands_to_sharpen,nodata,process,dtype):
        if process not in dir(self) or process!="nil":
            self.process = process
        else:
            raise ValueError("Must specify a process - if none requried, state 'nil'")
        if source not in ["DEA", "MPC"]:
            raise ValueError("Source must be either 'DEA' (Digital Earth Australia) or 'MPC' (Microsoft Planetary Computer)")
        self.source = source
        if isinstance(collections, str):
            self.collections = [collections]
        elif isinstance(collections, list) and all(isinstance(c, str) for c in collections):
            self.collections = collections
        else:
            raise ValueError("Collections must be a string or a list of strings")
        if not isinstance(resolution, (int, float)) or resolution <= 0:
            raise ValueError("Resolution must be a positive number")
        self.resolution = resolution
        if not (isinstance(bands_to_include, list) and all(isinstance(b, str) for b in bands_to_include)):
            raise ValueError("bands_to_include must be a list of band names (strings)")
        self.bands_to_include = bands_to_include
        if bands_to_resample in ["nearest","bilinear"]:
            self.bands_to_resample = bands_to_resample
        elif isinstance(bands_to_resample, dict):
            self.bands_to_resample = bands_to_resample
        else:
            raise ValueError('bands_to_resample must be a dict like {"band_name":{"ref":"reference_band_name","method":"resample method"}} or "nil"')
        if bands_to_sharpen == "nil":
            self.bands_to_sharpen = bands_to_sharpen
        elif isinstance(bands_to_sharpen, dict):
            self.bands_to_sharpen = bands_to_sharpen
        else:
            raise ValueError('bands_to_sharpen must be a dict like {"band_name":{"ref":"reference_band_name","method":"resample method"}} or "nil"')
        if nodata is None:
            raise ValueError("Must specify a nodata value")
        self.nodata = nodata
        if dtype is None:
            raise ValueError("Must specify a nodata value")
        self.dtype=dtype
    def make_product_dask(self,ds,region_manager,sub_region_code=None,compute=False):
        pds = None
        if self.process == "nil":
            pds = ds
        elif self.process == 'make_multiband':
            pds = self.make_multiband(ds,region_manager,sub_region_code)
        elif self.process == 'denoise_lee':
            pds = self.make_denoised(ds,region_manager,sub_region_code)
        else:
            pass
        if compute:
            return pds.compute()
        else:
            return pds

    def merge_clear_pixels(self):
        pass
    @dask.delayed
    def make_multiband(self,ds,region_manager,sub_region=None):
        # Get the polygon are to clip to
        geom = region_manager.get_polygon(sub_region,target_crs = ds.rio.crs.to_epsg())
        if self.bands_to_sharpen != "nil":
            ds = sharpen.apply_pansharpening(ds,self.bands_to_sharpen,reset_neg_val=1)
        ds = ds.astype(self.dtype)
        nodata_val = self.nodata
        for var in ds.data_vars:
            ds[var].rio.write_nodata(nodata_val, inplace=True)
        ds = ds.rio.clip([geom], drop=True)
        return(ds)
    @dask.delayed
    def make_denoised(self,ds,region_manager,sub_region=None):
        geom = region_manager.get_polygon(sub_region,target_crs = ds.rio.crs.to_epsg())
        pds = sen1.apply_lee_filter(ds)
        for var in ds.data_vars:
            ds[var].rio.write_nodata(self.nodata, inplace=True)
        pds = pds.rio.clip([geom], drop=True)
        return(pds)
