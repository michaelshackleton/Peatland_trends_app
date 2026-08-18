import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr
import dask
import dask.array as da


def sfim_pansharpen(pan, ms, sigma=1):
    """
    Purpose: Perform pan sharpening using smooth-filter intensity modulation
    """
    pan = pan.astype(np.float32)
    ms = ms.astype(np.float32)
    # Apply Gaussian filter to the PAN image to get the low-frequency component
    pan_low_freq = gaussian_filter(pan, sigma=sigma)
    # Initialize the result image
    pansharpened = np.zeros_like(ms)
    with np.errstate(divide='ignore', invalid='ignore'):
        pansharpened = np.where(pan_low_freq != 0, ms + (ms / pan_low_freq) * (pan - pan_low_freq), 0)
    return pansharpened
def hpf_pansharpen(pan, ms, sigma=1):
    """High-Pass Filter (HPF) pansharpening"""
    pan = pan.astype(np.float32)
    ms = ms.astype(np.float32)

    # Low-pass filtered PAN
    pan_low_freq = gaussian_filter(pan, sigma=sigma)
    high_pass = pan - pan_low_freq

    # Add high-pass to each MS band
    pansharpened = ms + high_pass
    return np.array(pansharpened)
import xarray as xr
import numpy as np
from scipy.ndimage import gaussian_filter
import dask.array as da

def apply_gaussian_filter_xarray(data, sigma=1):
    """Apply Gaussian filter to xarray DataArray with dask support"""
    if isinstance(data.data, da.Array):
        return xr.apply_ufunc(
            lambda x: gaussian_filter(x, sigma=sigma),
            data,
            dask='allowed'
        )
    else:
        return xr.apply_ufunc(
            lambda x: gaussian_filter(x, sigma=sigma),
            data
        )

def sfim_pansharpen_band(ms_band, pan_band, sigma=1):
    """
    SFIM pansharpening for a single band

    Parameters:
    -----------
    ms_band : xarray.DataArray
        Single multispectral band
    pan_band : xarray.DataArray
        Panchromatic/reference band
    sigma : float
        Standard deviation for Gaussian kernel

    Returns:
    --------
    xarray.DataArray
        Pansharpened band
    """
    with xr.set_options(keep_attrs=True):
        # Apply Gaussian filter to get low-frequency component
        pan_low_freq = apply_gaussian_filter_xarray(pan_band, sigma=sigma)

        # Apply SFIM algorithm
        ratio = xr.where(pan_low_freq != 0, ms_band / pan_low_freq, 0)
        modulation = ratio * (pan_band - pan_low_freq)
        result = ms_band + modulation

        # Update attributes
        result.attrs.update(ms_band.attrs)
        result.attrs['description'] = f'Pansharpened using SFIM (sigma={sigma})'

    return result

def hpf_pansharpen_band(ms_band, pan_band, sigma=1):
    """
    HPF pansharpening for a single band

    Parameters:
    -----------
    ms_band : xarray.DataArray
        Single multispectral band
    pan_band : xarray.DataArray
        Panchromatic/reference band
    sigma : float
        Standard deviation for Gaussian kernel

    Returns:
    --------
    xarray.DataArray
        Pansharpened band
    """
    with xr.set_options(keep_attrs=True):
        # Calculate high-pass component
        pan_low_freq = apply_gaussian_filter_xarray(pan_band, sigma=sigma)
        high_pass = pan_band - pan_low_freq

        # Add high-pass to ms band
        result = ms_band + high_pass

        # Update attributes
        result.attrs.update(ms_band.attrs)
        result.attrs['description'] = f'Pansharpened using HPF (sigma={sigma})'

    return result
def apply_pansharpening(dataset, bands_to_sharpen, sigma=1, reset_neg_val=None):
    """
    Apply pansharpening to multiple bands in a dataset based on configuration

    Parameters:
    -----------
    dataset : xarray.Dataset
        Input dataset containing all bands
    bands_to_sharpen : dict
        Dictionary specifying bands to sharpen, reference bands and methods
        Example: {"nbart_swir_2":{"ref":"nbart_nir_1","method":"hpf"}}
    sigma : float
        Standard deviation for Gaussian kernel

    Returns:
    --------
    xarray.Dataset
        Dataset with sharpened bands
    """
    # Create a copy of the dataset to modify
    result_ds = dataset.copy(deep=True)

    # Dictionary mapping method names to functions
    method_map = {
        "sfim": sfim_pansharpen_band,
        "hpf": hpf_pansharpen_band
    }

    # Process each band to sharpen
    for band_name, config in bands_to_sharpen.items():
        ref_band_name = config["ref"]
        method_name = config["method"].lower()

        if method_name not in method_map:
            raise ValueError(f"Unknown pansharpening method: {method_name}")

        # Select the appropriate function
        sharpen_func = method_map[method_name]

        # Check if we need to iterate over time dimension
        if 'time' in dataset[band_name].dims:
            # Process each time slice separately
            for t in dataset.time.values:
                # Extract single-time slices of the bands
                ms_slice = dataset[band_name].sel(time=t)
                ref_slice = dataset[ref_band_name].sel(time=t)
                # Apply pansharpening
                sharpened = sharpen_func(ms_slice, ref_slice, sigma=sigma)
                if reset_neg_val is not None:
                    sharpened = xr.where(sharpened < 0, reset_neg_val, sharpened)
                # Update the result dataset
                result_ds[band_name].loc[{"time": t}] = sharpened
        else:
            # No time dimension, process directly
            sharpened = sharpen_func(dataset[band_name], dataset[ref_band_name], sigma=sigma)
            if reset_neg_val is not None:
                sharpened = xr.where(sharpened < 0, reset_neg_val, sharpened)
            result_ds[band_name] = sharpened

    return result_ds
