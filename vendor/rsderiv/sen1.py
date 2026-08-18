import cv2
import numpy as np


def lee_filter_cv2(ds, var_name, window_size=7):
    """
    Apply Lee filter for speckle reduction using cv2

    Parameters:
    -----------
    ds : xarray.Dataset
        Input xarray dataset
    var_name : str
        Variable name to filter
    window_size : int
        Size of filter window (must be odd)

    Returns:
    --------
    xarray.Dataset
        Dataset with filtered variable
    """
    # Create output dataset
    ds_filtered = ds.copy(deep=True)
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    # Process each time slice
    if 'time' in ds.dims:
        # Iterate over indices instead of time values
        for t in range(len(ds.time)):
            # Get the data as numpy array using index
            img_array = ds[var_name].isel(time=t).values.astype(np.float32)

            # Handle NaN values by replacing with 0 (for processing)
            mask = np.isnan(img_array)
            img_valid = np.where(mask, 0, img_array)

            # Calculate image statistics with cv2
            img_mean = cv2.blur(img_valid, (window_size, window_size))
            img_mean_sq = cv2.blur(img_valid**2, (window_size, window_size))
            img_var = img_mean_sq - img_mean**2

            # Avoid division by zero or negative variance
            img_var = np.maximum(img_var, 0.0001)

            # Calculate global variance (from non-NaN values)
            valid_data = img_valid[~mask]
            if len(valid_data) > 0:
                global_var = np.var(valid_data)
            else:
                global_var = 0.0001

            # Lee filter
            weights = img_var / (img_var + global_var)
            output = img_mean + weights * (img_valid - img_mean)

            # Restore NaN values in the original positions
            output = np.where(mask, np.nan, output)

            # Replace the data in the dataset
            ds_filtered[var_name][t] = output
    else:
        img_array = ds[var_name].values.astype(np.float32)
        # Handle NaN values by replacing with 0 (for processing)
        mask = np.isnan(img_array)
        img_valid = np.where(mask, 0, img_array)

        # Calculate image statistics with cv2
        img_mean = cv2.blur(img_valid, (window_size, window_size))
        img_mean_sq = cv2.blur(img_valid**2, (window_size, window_size))
        img_var = img_mean_sq - img_mean**2

        # Avoid division by zero or negative variance
        img_var = np.maximum(img_var, 0.0001)

        # Calculate global variance (from non-NaN values)
        valid_data = img_valid[~mask]
        if len(valid_data) > 0:
            global_var = np.var(valid_data)
        else:
            global_var = 0.0001

        # Lee filter
        weights = img_var / (img_var + global_var)
        output = img_mean + weights * (img_valid - img_mean)

        # Restore NaN values in the original positions
        output = np.where(mask, np.nan, output)

        # Replace the data in the dataset
        ds_filtered[var_name] = output

    # Add processing info
    ds_filtered.attrs['processing'] = f'Lee filter (window size {window_size})'

    return ds_filtered

# Function to apply filter to multiple variables
def apply_lee_filter(ds, var_names=['vh', 'vv'], window_size=7):
    """
    Apply Lee filter to multiple variables in dataset
    """
    result = ds.copy(deep=True)

    for var in var_names:
        if var in ds:
            print(f"Applying Lee filter to {var}...")
            result = lee_filter_cv2(result, var, window_size)

    return result
