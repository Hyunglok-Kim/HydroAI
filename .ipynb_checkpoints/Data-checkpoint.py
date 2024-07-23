import numpy as np
import pandas as pd
import time
import atexit
import platform
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from functools import partial
from tqdm import tqdm
from scipy.spatial import cKDTree
from datetime import datetime, timedelta
import os
import glob

import netCDF4
from netCDF4 import Dataset
from tabulate import tabulate
import h5py

import matplotlib.pyplot as plt

from pyhdf.SD import SD, SDC

if platform.system() == 'Darwin':  # macOS
    import multiprocessing as mp
    from multiprocessing import Pool
else:  # assume Linux or other Unix-like system
    import multiprocess as mp
    from multiprocess import Pool
    
#def clear_cache():
#    load_data("", "", 0, clear_cache=True)
#    print("Cache cleared")
#atexit.register(clear_cache)

def create_3d_object_array(x, y, z):
    """
    Create a 3D NumPy array that can store list data, with the specified shape (x, y, z).
    
    Parameters:
    x (int): Size of the first dimension.
    y (int): Size of the second dimension.
    z (int): Size of the third dimension.
    
    Returns:
    np.ndarray: 3D NumPy array of shape (x, y, z) with dtype=object.
    """
    # Create an empty array with the given shape and dtype=object
    if z == 0:
        obj_array = np.empty((x, y), dtype=object)
        # Initialize each element to an empty list
        for i in range(x):
            for j in range(y):
                    obj_array[i, j] = []
    else:
        obj_array = np.empty((x, y, z), dtype=object)
        # Initialize each element to an empty list
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    obj_array[i, j, k] = []
    
    return obj_array

def object_array_to_np(obj_data, target_np_array):
    obj_shape = obj_data.shape
    
    for ii in range(obj_shape[0]):
        for jj in range(obj_shape[1]):
            for kk in range(obj_shape[2]):
                target_np_array[ii,jj,kk] = np.nanmean(obj_data[ii,jj,kk])
    return target_np_array
    
def create_3d_np_array(x, y, z, fill_value = np.nan):
    """
    Create a 3D NumPy array with the specified shape (x, y, z).
    
    Parameters:
    x (int): Size of the first dimension.
    y (int): Size of the second dimension.
    z (int): Size of the third dimension.
    
    Returns:
    np.ndarray: 3D NumPy array of shape (x, y, z).
    """
    if z == 0:
        return np.full((x, y), fill_value)
    else:
        return np.full((x, y, z), fill_value)
    
def load_data(input_fp, file_name, engine="c", clear_cache=False):
    
    if not hasattr(load_data, 'cache'):
        load_data.cache = {}
        
    start_time = time.time()
    
    if clear_cache:
        load_data.cache.clear()
        print("Cache cleared")
        return [], []
    
    if file_name not in load_data.cache:
        try:
            if engine == "pyarrow":
                load_data.cache[file_name] = pd.read_csv(input_fp + file_name, engine="pyarrow")
            else:
                load_data.cache[file_name] = pd.read_csv(input_fp + file_name)
            

        except FileNotFoundError:
            print(f"File not found: {input_fp + file_name}")
            flag = 0
            return [], []
    
    end_time = time.time()
    print(f"Data Load Time Taken:({file_name}) {end_time - start_time:.4f} seconds")
    data = load_data.cache[file_name]
    columns = data.columns
    
    return data, columns

#def mode_function(x):
#    return x.mode().iloc[0]

def mode_function(x):
    mode_val = x.mode()
    if not mode_val.empty:
        return mode_val.iloc[0]
    return np.nan

def gini_simpson(values):
    if values.empty or values.isna().all():
        return np.nan
    _, counts = np.unique(values.dropna(), return_counts=True)
    probabilities = counts / counts.sum()
    gini_simpson_index = 1 - np.sum(probabilities**2)
    return gini_simpson_index

def magnify_VAR(lon_input, lat_input, VAR, mag_factor):
    # Rescale lon and lat using bilinear interpolation (order=1)
    m_lon = zoom(lon_input, mag_factor, order=1)
    m_lat = zoom(lat_input, mag_factor, order=1)
    m_values = zoom(VAR, mag_factor, order=0)  # Nearest neighbor interpolation
    #print("After magnification (lat, lon, value shape):", m_lat.shape, m_lon.shape, m_values.shape)
    return m_lon, m_lat, m_values

def Resampling_test(lon_target, lat_target, lon_input, lat_input, VAR, sampling_method, agg_method='mean', mag_factor=3):
    #--------------------------BEGIN NOTE------------------------------%
    # University of Virginia
    # USDA
    # HydroAI lab in GIST
    #--------------------------END NOTE--------------------------------%
    # ARGUMENTS:
    # lat_target/ lon_target : Target frame lat/lon data (m x n arrays)
    # lat_input / lon_input : Satellite lat/lon data (m' x n' arrays)
    # (NOTE: lat(i,1)>lat(i+1,1) (1<=i<=(size(lat_main,1)-1))
    #        lon(1,i)<lon(1,i+1) (1<=i<=(size(lon_main,2)-1)) )
    #
    # VAR : Satellite's variable (m' x n' array)
    # method: Method for resampling: (e.g., 'nearest')
    #
    # sampling_method: determines the interpolation method or algorithm to be used
    #             (e.g., linear, nearest, zero, slinear, quadratic, cubic)
    # agg_method: determines the interpolation order to use when resizing the input array
    #             (e.g., mean, median, mode, min, max)
    #
    # DESCRIPTION:
    # This code resampled earth coordinates of the specified domain for 
    # any "target" projection
    #
    # REVISION HISTORY: 
    # 2 Jul 2020 Hyunglok Kim; initial specification in Matlab
    # 16 May 2023 Hyunglok Kim; converted to Python code
    # 17 Apr 2024 Hyunglok Kim; algorithm updated with cKDTree algorithm
    #-----------------------------------------------------------------%

    s_target = lat_target.shape
    s_input = lat_input.shape

    if mag_factor > 1:
        lon_input, lat_input, VAR = magnify_VAR(lon_input, lat_input, VAR, mag_factor)

    # Flatten the coordinate arrays
    coords_input = np.column_stack([lat_input.ravel(), lon_input.ravel()])
    coords_target = np.column_stack([lat_target.ravel(), lon_target.ravel()])

    # Build KDTree from target coordinates
    tree = cKDTree(coords_target)

    # Query KDTree to find the nearest target index for each input coordinate
    distances, indices = tree.query(coords_input)

    # Create a DataFrame for aggregation
    df = pd.DataFrame({
        'values': VAR.ravel(),
        'indices': indices  # This maps each input value to its nearest target cell
    })

    # Aggregate the data
    if agg_method == 'mode':
        agg_values = df.groupby('indices')['values'].agg(lambda x: pd.Series.mode(x).iloc[0])
    elif agg_method == 'count':
        agg_values = df.groupby('indices')['values'].count()  # Correct usage of count
    elif agg_method == 'gini_simpson':
        agg_values = df.groupby('indices')['values'].agg(gini_simpson)  # Using the custom function
    else:
        agg_values = df.groupby('indices')['values'].agg(agg_method)

    # Prepare the result array
    VAR_r = np.full(lat_target.shape, np.nan)
    for idx, value in agg_values.items():
        np.put(VAR_r, idx, value)  # Assign aggregated values to their corresponding indices in the result array

    return VAR_r
    
def Resampling(lon_target, lat_target, lon_input, lat_input, VAR, sampling_method='nearest', agg_method='mean', mag_factor=2):
    '''
    --------------------------BEGIN NOTE------------------------------%
     University of Virginia
     USDA
     HydroAI lab in GIST
    --------------------------END NOTE--------------------------------%
     ARGUMENTS:
     lat_target/ lon_target : Target frame lat/lon data (m x n arrays)
     lat_input / lon_input : Satellite lat/lon data (m' x n' arrays)
     (NOTE: lat(i,1)>lat(i+1,1) (1<=i<=(size(lat_main,1)-1))
            lon(1,i)<lon(1,i+1) (1<=i<=(size(lon_main,2)-1)) )
    
     VAR : Satellite's variable (m' x n' array)
     method: Method for resampling: (e.g., 'nearest')
    
     sampling_method: determines the interpolation method or algorithm to be used
                 (e.g., linear, nearest, zero, slinear, quadratic, cubic)
     agg_method: determines the interpolation order to use when resizing the input array
                 (e.g., mean, median, mode, min, max)
    
     DESCRIPTION:
     This code resampled earth coordinates of the specified domain for 
     any "target" projection
    
     REVISION HISTORY: 
     2 Jul 2020 Hyunglok Kim; initial specification in Matlab
     16 May 2023 Hyunglok Kim; converted to Python code
     19 Apr 2024 Hyunglok Kim; Gini-simpson index added
     23 May 2024 Hyunglok Kim; Resampling condition added
    -----------------------------------------------------------------%
    '''
    if np.array_equal(lon_target, lon_input): # do not resample it if a target and input data are the equal projection/resolution
       return VAR
    else:
        if mag_factor > 1:
            lon_input, lat_input, VAR = magnify_VAR(lon_input, lat_input, VAR, mag_factor)
    
        def resample_agg(lon_target, lat_target, lon_input, lat_input, VAR, sampling_method, agg_method):
            nan_frame = np.empty(lat_target.shape)
            nan_frame[:] = np.nan
            VAR_r = nan_frame
    
            valid_data = (~np.isnan(VAR)) & (lat_input <= np.max(lat_target[:,0])) & \
                         (lat_input > np.min(lat_target[:,0])) & (lon_input < np.max(lon_target[0,:])) & \
                         (lon_input >= np.min(lon_target[0,:]))
    
            valid_value = VAR[valid_data]
            t_lat = lat_input[valid_data]
            t_lon = lon_input[valid_data]
    
            f_lat = interp1d(lat_target[:,0], np.arange(lat_target.shape[0]), kind=sampling_method, bounds_error=False)
            f_lon = interp1d(lon_target[0,:], np.arange(lon_target.shape[1]), kind=sampling_method, bounds_error=False)
    
            t_lat_index = f_lat(t_lat)
            t_lon_index = f_lon(t_lon)
    
            index_array = np.ravel_multi_index([t_lat_index.astype(int), t_lon_index.astype(int)], lat_target.shape)
            nan_valid = np.isnan(np.sum([t_lat_index, t_lon_index], axis=0))
            valid_value = valid_value[~nan_valid]
            df = pd.DataFrame({'idx': index_array[~nan_valid], 'val': valid_value})
    
            # Aggregate the data       
            if agg_method == 'mode':
                agg_values = df.groupby('idx')['val'].apply(lambda x: pd.Series.mode(x).iloc[0])
            elif agg_method == 'count':
                agg_values = df.groupby('idx')['val'].count()  # Correct usage of count
            elif agg_method == 'gini_simpson':
                agg_values = df.groupby('idx')['val'].agg(gini_simpson)  # Using the custom function
            else:
                agg_values = getattr(df.groupby('idx')['val'], agg_method)()
                
            VAR_r[np.unravel_index(agg_values.index.values, VAR_r.shape)] = agg_values.values
    
            return VAR_r
    
        if lat_target.shape == lat_input.shape and np.all(lat_target == lat_input) and np.all(lon_target == lon_input):
            print('Resampling is not required.')
            VAR_r = VAR
        else:
            VAR_r = resample_agg(lon_target, lat_target, lon_input, lat_input, VAR, sampling_method, agg_method)
        return VAR_r

def process_var(i, lon_target, lat_target, lon_input, lat_input, data, sampling_method,agg_method, mag_factor):
    #print(i)
    VAR = data[:,:,i]
    result = Resampling(lon_target, lat_target, lon_input, lat_input, VAR, sampling_method, agg_method, mag_factor)
    return result

def Resampling_forloop(lon_target, lat_target, lon_input, lat_input, VAR, sampling_method='nearest', agg_method='mean', mag_factor=3):
    
    m, n = lat_target.shape  # Get the dimensions from lat_target
    # Initialize results array
    results = np.empty((m, n, VAR.shape[2]))
    
    for i in tqdm(range(0, VAR.shape[2])):
        t = Resampling(lon_target, lat_target, lon_input, lat_input, VAR[:,:,i],sampling_method, agg_method, mag_factor)
        results[:,:,i] = t

    return results

def Resampling_parallel(lon_target, lat_target, lon_input, lat_input, VAR, sampling_method='nearest',agg_method='mean', mag_factor=3):

    # Create a partial function with the arguments that don't change
    partial_process_var = partial(process_var, lon_target=lon_target, lat_target=lat_target,
                                  lon_input=lon_input, lat_input=lat_input, data=VAR, sampling_method=sampling_method, agg_method=agg_method, mag_factor=mag_factor)
    m, n = lat_target.shape  # Get the dimensions from lat_target

    # Initialize results array
    results = np.empty((m, n, VAR.shape[2]))

    with Pool(8) as p:
        results_list = p.map(partial_process_var, range(VAR.shape[2]))

    for i, result in enumerate(results_list):
        results[:,:,i] = result

    return results

def moving_average_3d(data, window_size, mode='+-', min_valid_fraction=0.3):
    m, n, z = data.shape
    
    if mode == 'past':
        padding = (window_size - 1, 0)  # Pad only on the left
    elif mode == 'post':
        padding = (0, window_size - 1)  # Pad only on the right
    elif mode == '+-':
        padding = (window_size, window_size)  # Pad on both sides
    else:
        raise ValueError("Mode should be 'past', 'post', or '+-'")
    
    # Pad the data with NaN values to handle the edges
    padded_data = np.pad(data, ((0, 0), (0, 0), padding), mode='constant', constant_values=np.nan)
    
    # Create an array to store the moving averaged values
    moving_averaged = np.full((m, n, z), np.nan)  # Initialize with NaNs
    
    # Minimum number of valid points required
    min_valid_points = int(window_size * min_valid_fraction)
    
    # Calculate the moving average for each row and pixel
    for k in tqdm(range(z), desc="Calculating moving average"):
        if mode == 'past':
            if k < window_size:
                window_data = padded_data[:, :, :k+1]
            else:
                window_data = padded_data[:, :, k-window_size+1:k+1]
        elif mode == 'post':
            window_data = padded_data[:, :, k:k+window_size]
        elif mode == '+-':
            start = k - window_size
            end = k + window_size + 1
            window_data = padded_data[:, :, start:end]
        
        valid_counts = np.sum(~np.isnan(window_data), axis=2)
        
        with np.errstate(invalid='ignore'):  # Ignore warnings due to NaNs
            moving_averaged[:, :, k] = np.where(valid_counts >= min_valid_points, np.nanmean(window_data, axis=2), np.nan)
    
    return moving_averaged


#def moving_average_3d(data, window_size):
#    m, n, z = data.shape
#    padding = window_size // 2  # Number of elements to pad on each side
#    
#    # Pad the data with NaN values to handle the edges
#    padded_data = np.pad(data, ((0, 0), (0, 0), (padding, padding)), mode='constant', constant_values=np.nan)
#    
#    # Create an array to store the moving averaged values
#    moving_averaged = np.zeros((m, n, z))
#    
#    # Calculate the moving average for each row and pixel
#    # This code can be modified not to average already "moving averaged" values
#    # This code can be modified not to average if there is less than certain numbers of valid points
#    for k in range(z):
#        moving_averaged[:, :, k] = np.nanmean(padded_data[:, :, k:k+window_size], axis=2)
#    
#    return moving_averaged

def find_closest_index_old(longitudes, latitudes, point):
    lon_lat = np.c_[longitudes.ravel(), latitudes.ravel()]
    tree = cKDTree(lon_lat)
    dist, idx = tree.query(point, k=1)
    return np.unravel_index(idx, latitudes.shape)

def is_uniform(array, axis):
    """
    Check if all rows or columns in the array are the same.

    Parameters:
    array (np.ndarray): Input array.
    axis (int): Axis to check for uniformity. 0 for columns, 1 for rows.

    Returns:
    np.ndarray: Boolean array indicating uniformity along the specified axis.
    """
    if axis == 1:  # Check row-wise
        return np.all(array == array[0, :][None, :], axis=1)
    elif axis == 0:  # Check column-wise
        return np.all(array == array[:, 0][:, None], axis=0)
    else:
        raise ValueError("Axis must be 0 (columns) or 1 (rows).")
        
def find_closest_index(lon_2d, lat_2d, coord):
    """
    Find the closest index in a 2D grid of longitude and latitude values to a given coordinate.

    Parameters:
    lon_2d (np.ndarray): 2D array of longitude values.
    lat_2d (np.ndarray): 2D array of latitude values.
    coord (tuple): A tuple containing the longitude and latitude of the target coordinate (lon_value, lat_value).

    Returns:
    tuple: A tuple containing the indices (lat_idx, lon_idx) of the closest grid point.

    Explanation:
    The function first checks if the rows of `lon_2d` are uniform and the columns of `lat_2d` are uniform.
    If both conditions are met, it indicates that the grid is uniform, and the process speed is greatly increased
    due to direct indexing. If the grids are not uniform, the function uses a KDTree for nearest-neighbor search,
    which is computationally more intensive.

    REVISION HISTORY: 
    2 June 2024 Hyunglok Kim; initial specification
    """
    
    lon_value, lat_value = coord

    if np.all(is_uniform(lon_2d, 1)) and np.all(is_uniform(lat_2d, 0)):
        # Case when lon_2d's rows are uniform and lat_2d's columns are uniform
        lon_unique = lon_2d[0, :]
        lat_unique = lat_2d[:, 0]
        lon_idx = (np.abs(lon_unique - lon_value)).argmin()
        lat_idx = (np.abs(lat_unique - lat_value)).argmin()
    else:
        # Case when lon_2d and lat_2d are not uniform
        lon_flat = lon_2d.flatten()
        lat_flat = lat_2d.flatten()
        
        coordinates = np.vstack((lon_flat, lat_flat)).T
        tree = cKDTree(coordinates)
        
        dist, idx = tree.query(coord)
        lat_idx, lon_idx = np.unravel_index(idx, lon_2d.shape)
    
    lat_idx, lon_idx = int(lat_idx), int(lon_idx)
    return lat_idx, lon_idx

    
def extract_region_from_data(longitude, latitude, X, bounds):
    """
    Create a subset of a 3D array based on given latitude and longitude bounds.
    
    Args:
    - X: The 3D array to subset. The first two dimensions should correspond to latitude and longitude.
    - latitude: 2D array of latitude values corresponding to the first dimension of X.
    - longitude: 2D array of longitude values corresponding to the second dimension of X.
    - bounds: Tuple of (lon_min, lon_max, lat_min, lat_max).
    
    Returns:
    - A subset of X corresponding to the specified bounds.
    """
    lon_min, lon_max, lat_min, lat_max = bounds
    
    # Find indices for the bounding box
    lat_indices = np.where((latitude >= lat_min) & (latitude <= lat_max))
    lon_indices = np.where((longitude >= lon_min) & (longitude <= lon_max))

    # Find the minimum and maximum indices to slice the array
    lat_min_idx, lat_max_idx = min(lat_indices[0]), max(lat_indices[0])
    lon_min_idx, lon_max_idx = min(lon_indices[1]), max(lon_indices[1])

    # Subset the array
    subset = X[lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1, :]
    
    return subset

def average_over_space(X):
    """
    Compute the average over the first two dimensions in a 3D array.
    
    Args:
    - X: The 3D array to compute the average on. The first two dimensions are averaged.
    
    Returns:
    - A 1D array of shape (Z,) representing the average over the first two dimensions for each layer in the third dimension.
    """
    # Compute the mean over the first two dimensions (latitude and longitude)
    mean_values = np.nanmean(X, axis=(0, 1))
    
    return mean_values

def get_file_list(directory_path, file_extension, recursive=True, filter_strs=None):
    """
    Lists all files in the specified directory and its subdirectories (if recursive is True)
    with the given file extension. Optionally filters files to include only those containing any of the specified substrings.
    Additionally, sorts the resulting file paths in ascending order.

    Args:
        directory_path (str): The path to the directory where the files are located.
        file_extension (str): The file extension to search for.
        recursive (bool): Whether to search files recursively in subdirectories.
        filter_strs (list of str, optional): List of substrings that must be included in the filenames.

    Returns:
        list: A sorted list of full file paths matching the given file extension and containing any of the filter strings (if provided).
    """
    # Ensure the file extension starts with a dot
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension

    # Construct the search pattern
    if recursive:
        pattern = os.path.join(directory_path, '**', f'*{file_extension}')
    else:
        pattern = os.path.join(directory_path, f'*{file_extension}')

    # Get a list of all files matching the pattern
    file_paths = glob.glob(pattern, recursive=recursive)

    # Filter files to include only those containing any of the filter_strs, if provided
    if filter_strs:
        filtered_paths = []
        for file_path in file_paths:
            base_name = os.path.basename(file_path)
            if any(substring.strip("'\"") in base_name for substring in filter_strs):
                filtered_paths.append(file_path)
        file_paths = filtered_paths

    # Sort the file paths in ascending order
    file_paths = sorted(file_paths)

    return file_paths


# Example usage
#directory_path = '/data/X'
#file_extension = 'hdf'  # Example file format
# Get all text files
# all_txt_files = get_file_list(directory, file_ext)
# Get text files that include "abs", "2021", or "report" in the filename
# filtered_txt_files = get_file_list(directory, file_ext, filter_strs=["abs","report"])
# or
# filtered_txt_files = get_file_list(directory, file_ext, filter_strs=['abs'])

### netcdf modules ###
def create_netcdf_file(nc_file, longitude, latitude, time_arg='doy', **data_vars):
    """
    Creates a NetCDF file from the provided data arrays and latitude/longitude grids.

    Args:
        nc_file (str): Path to the output NetCDF file.
        latitude (np.array): 2D array of latitude values.
        longitude (np.array): 2D array of longitude values.
        data_vars (dict): Dictionary of 3D data arrays to include in the NetCDF file.
        time_arg (str): Name of time axis.

    Returns:
        None
    """
    # Create a new NetCDF file
    nc_data = netCDF4.Dataset(nc_file, 'w')

    # Define the dimensions
    rows, cols = latitude.shape
    # Assuming all data variables have the same 'time' dimension size
    if next(iter(data_vars.values())).ndim == 1:
        time = next(iter(data_vars.values())).shape[0]
    elif next(iter(data_vars.values())).ndim == 2:
        time = 1
    else:
        time = next(iter(data_vars.values())).shape[2]

    # Create dimensions in the NetCDF file
    nc_data.createDimension('latitude', rows)
    nc_data.createDimension('longitude', cols)
    nc_data.createDimension(time_arg, time)

    # Create latitude and longitude variables
    lat_var = nc_data.createVariable('latitude', 'f4', ('latitude', 'longitude'))
    lon_var = nc_data.createVariable('longitude', 'f4', ('latitude', 'longitude'))

    # Assign data to the latitude and longitude variables
    lat_var[:] = latitude
    lon_var[:] = longitude

    # Create variables and assign data for each item in data_vars
    for var_name, var_data in data_vars.items():
        # Create variable in NetCDF file
        if var_data.ndim == 1:
            if isinstance(var_data[0], np.int64):
                nc_var = nc_data.createVariable(var_name, 'i4', (time_arg, ))
            else:
                nc_var = nc_data.createVariable(var_name, 'f4', (time_arg, ))
        else:  
            nc_var = nc_data.createVariable(var_name, 'f4', ('latitude', 'longitude', time_arg))
        # Assign data to the variable
        nc_var[:] = var_data

    # Close the NetCDF file
    nc_data.close()

    print(f"NetCDF file {nc_file} created successfully.")

# Example usage
#hData.create_netcdf_file(
#        nc_file    = nc_file_name,
#        latitude   = domain_lat,
#        longitude  = domain_lon,
#        study_dates = study_dates,                   # 1D list of integer such as [20240101, 20240102, ...]
#        Resampled_SMOS_SM    = Resampled_SMOS_SM,
#        Resampled_SMOS_SM_QC = Resampled_SMOS_SM_QC,
#        time_arg = 'dates_yymmdd'                       # Default argument is 'doy'. This argument means name of time axis.
#        )

def get_nc_variable_names_units(nc_file_path):
    """
    Get a list of variable names, a corresponding list of their units, 
    and a corresponding list of their long names from a NetCDF file.
    Additionally, print this information in a table format.

    :param nc_file_path: Path to the NetCDF file.
    :return: A list of variable names, a list of units for these variables,
             and a list of long names for these variables.
    """
    variable_names = []
    variable_units_list = []
    variable_long_names_list = []
    
    with Dataset(nc_file_path, 'r') as nc:
        # Extract the list of variable names
        variable_names = list(nc.variables.keys())
        
        # Create lists that contain the units and long names for each variable
        for var_name in variable_names:
            try:
                # Try to get the 'units' attribute for the variable
                variable_units_list.append(nc.variables[var_name].units)
            except AttributeError:
                # If the variable doesn't have a 'units' attribute, append None
                variable_units_list.append(None)
            
            try:
                # Try to get the 'long_name' attribute for the variable
                variable_long_names_list.append(nc.variables[var_name].long_name)
            except AttributeError:
                # If the variable doesn't have a 'long_name' attribute, append None
                variable_long_names_list.append(None)
    
    # Prepare the data for tabulation
    table_data = zip(variable_names, variable_long_names_list, variable_units_list)

    # Print the results in a table-like format
    print(tabulate(table_data, headers=["Name", "Long Name", "Units"], tablefmt="grid"))

    return variable_names, variable_units_list, variable_long_names_list

def get_variable_from_nc(nc_file_path, variable_name, layer_index='all', flip_data=False):
    """
    Extract a specific layer (if 3D), the entire array (if 2D or 1D), or the value (if 0D) of a variable
    from a NetCDF file and return it as a NumPy array, with fill values replaced by np.nan.

    :param nc_file_path: Path to the NetCDF file.
    :param variable_name: Name of the variable to extract.
    :param layer_index: The index of the layer to extract if the variable is 3D. Default is 0.
    :return: NumPy array or scalar of the specified variable data, with np.nan for fill values.
    """
    with Dataset(nc_file_path, 'r') as nc:
        # Check if the variable exists in the NetCDF file
        if variable_name in nc.variables:
            variable = nc.variables[variable_name]

            # Extract data based on the number of dimensions
            if variable.ndim == 4:
                # Extract the specified layer for 3D variables
                if layer_index == 'all':
                    data = variable[:, 0, :, :]
                else:
                    data = variable[layer_index, 0, :, :]
            elif variable.ndim == 3:
                # Extract the specified layer for 3D variables
                if layer_index == 'all':
                    data = variable[:, :, :]
                else:
                    data = variable[layer_index, :, :]
            elif variable.ndim == 2:
                # Extract all data for 2D variables
                data = variable[:, :]
            elif variable.ndim == 1:
                # Extract all data for 1D variables
                data = variable[:]
            elif variable.ndim == 0:
                # Extract scalar value for 0D variables
                data = variable.getValue()
            else:
                raise ValueError(f"Variable '{variable_name}' has unsupported number of dimensions: {variable.ndim}.")

            # Handle fill values (mask to NaN if necessary)
            if isinstance(data, np.ma.MaskedArray):
                fill_value = np.nan if np.issubdtype(data.dtype, np.floating) else -9999
                data = data.filled(fill_value)
                data = np.where(data == -9999, np.nan, data)  # Replace fill_value with NaN for integer arrays
            
            # Flip the data upside down if it's 2D or 3D (common in geographical data to match orientation)
            if flip_data and variable.ndim in [2, 3]:
                data = np.flipud(data)

            return data
        else:
            raise ValueError(f"Variable '{variable_name}' does not exist in the NetCDF file.")

### h4 modules ###
def inspect_hdf4_file(input_file):
    """
    Inspects the contents of an HDF4 file, printing out the names of datasets.

    Args:
    input_file (str): The path to the HDF4 file to inspect.
    """
    try:
        # Open the HDF4 file in read mode
        hdf = SD(input_file, SDC.READ)
        print("Contents of the HDF4 file:")
        datasets = hdf.datasets()
        for name, info in datasets.items():
            # Print basic information about each dataset
            print(f"Dataset: {name}")
            print(f" - Dimensions: {info[0]}")
            print(f" - Type: {info[3]}")
            # Optionally print the type of data stored
            data = hdf.select(name)
            print(f" - Data Type: {data.info()[3]}")
            data.endaccess()
    except Exception as e:
        print(f"Failed to read HDF4 file: {e}")

def read_hdf4_variable(input_file, variable_name):
    """
    Reads a specified variable from an HDF4 file.

    Args:
    input_file (str): The path to the HDF4 file.
    variable_name (str): The name of the variable to read.

    Returns:
    numpy.ndarray: The data of the specified variable, or None if an error occurs.
    """
    try:
        # Open the HDF4 file in read mode
        hdf = SD(input_file, SDC.READ)
        # Select the dataset by the variable name
        dataset = hdf.select(variable_name)
        # Read the data from the dataset
        data = dataset[:]
        # Clean up: end access to the dataset
        dataset.endaccess()
        # Return the data array
        return data

    except Exception as e:
        print(f"Failed to read '{variable_name}' from HDF4 file {input_file}: {e}")
        return None
### ------------------------------------------- ###

### h5 modules ###
def get_h5_variable_names_units(h5_file_path):
    """
    Get a list of variable names, a corresponding list of their units,
    and a corresponding list of their long names from an HDF5 file.
    Additionally, print this information in a table format.

    :param h5_file_path: Path to the HDF5 file.
    :return: A list of variable names, a list of units for these variables,
             and a list of long names for these variables.
    """
    variable_names = []
    variable_units_list = []
    variable_long_names_list = []

    with h5py.File(h5_file_path, 'r') as file:
        def extract_attributes(name, obj):
            if isinstance(obj, h5py.Dataset):
                variable_names.append(name)
                # Try to get the 'units' attribute for the variable
                units = obj.attrs.get('units', None)
                variable_units_list.append(units)
                # Try to get the 'long_name' attribute for the variable
                long_name = obj.attrs.get('long_name', None)
                variable_long_names_list.append(long_name)
        
        # Recursively visit all items in the HDF5 file and extract attributes
        file.visititems(extract_attributes)
    
    # Prepare the data for tabulation
    table_data = zip(variable_names, variable_long_names_list, variable_units_list)

    # Print the results in a table-like format
    print(tabulate(table_data, headers=["Name", "Long Name", "Units"], tablefmt="grid"))

    return variable_names, variable_units_list, variable_long_names_list

def get_variable_from_h5(h5_file_path, variable_name, layer_index='all', flip_data=False):
    """
    Extract a specific layer (if 3D), the entire array (if 2D or 1D), or the value (if 0D) of a variable
    from an HDF5 file and return it as a NumPy array, with fill values replaced by np.nan.

    :param h5_file_path: Path to the HDF5 file.
    :param variable_name: Full path name of the variable to extract.
    :param layer_index: The index of the layer to extract if the variable is 3D. Default is 'all'.
    :param flip_data: Boolean to indicate if the data should be flipped upside down. Default is False.
    :return: NumPy array or scalar of the specified variable data, with np.nan for fill values.
    """
    with h5py.File(h5_file_path, 'r') as file:
        # Check if the variable exists in the HDF5 file
        if variable_name in file:
            variable = file[variable_name]
            
            # Extract data based on the number of dimensions
            data = variable[:]
            if variable.ndim == 4:
                if layer_index == 'all':
                    data = data[:, 0, :, :]
                else:
                    data = data[layer_index, 0, :, :]
            elif variable.ndim == 3:
                if layer_index == 'all':
                    data = data[:, :, :]
                else:
                    data = data[layer_index, :, :]
            elif variable.ndim == 2:
                data = data[:, :]
            elif variable.ndim == 1:
                data = data[:]
            elif variable.ndim == 0:
                data = data[()]
            else:
                raise ValueError(f"Variable '{variable_name}' has unsupported number of dimensions: {variable.ndim}.")
            
            # Handle fill values (mask to NaN if necessary)
            if isinstance(data, np.ma.MaskedArray):
                fill_value = np.nan if np.issubdtype(data.dtype, np.floating) else -9999
                data = data.filled(fill_value)
                data = np.where(data == -9999, np.nan, data)  # Replace fill_value with NaN for integer arrays
            
            # Flip the data upside down if it's 2D or 3D (common in geographical data to match orientation)
            if flip_data and variable.ndim in [2, 3]:
                data = np.flipud(data)
            
            return data
        else:
            raise ValueError(f"Variable '{variable_name}' does not exist in the HDF5 file.")

#---- UTC to LT ----#
# Function to calculate local time for a given longitude
def calculate_local_time(utc_time, longitude):
    # Calculate the time difference in hours
    time_difference = longitude / 15.0
    # Calculate local time
    local_time = utc_time + timedelta(hours=time_difference)
    return local_time

def doy_to_yearyyyymmdd(year, doy):
    # Create a date object for the first day of the given year
    start_date = datetime(year, 1, 1)
    # Add the DOY to the start date to get the correct date
    target_date = start_date + timedelta(days=doy - 1)
    # Format the output as yearmmdd
    formatted_date = target_date.strftime('%Y%m%d')
    return formatted_date

def days_in_year(year):
    """
    Return the number of days in a given year.
    
    :param year: The year to check.
    :return: 366 if the year is a leap year, otherwise 365.
    """
    if (year % 4 == 0):
        if (year % 100 == 0):
            if (year % 400 == 0):
                return 366
            else:
                return 365
        else:
            return 366
    else:
        return 365

def UTC_to_LT(data_FP, target_local_time, lon, year, doy, var_name, layer_index=0):
    reference_time = datetime(2000, 1, 1, 3, 0, 0)
    t_nc_file_paths = get_file_list(data_FP, 'nc4', filter_strs=[doy_to_yearyyyymmdd(year, doy-1), doy_to_yearyyyymmdd(year, doy), doy_to_yearyyyymmdd(year, doy+1)])
    t_var_LT_combined = np.full((lon.shape), np.nan)
    
    for i in tqdm(t_nc_file_paths, desc='Processing files', unit='file'):
        t_var = get_variable_from_nc(i, var_name, layer_index=layer_index, flip_data='False')
        t_UTC_time = get_variable_from_nc(i, 'time', layer_index=0, flip_data='False')
        
        # Convert observed minutes to a datetime object
        t_UTC_time = reference_time + timedelta(minutes=t_UTC_time[0])
        t_local_times = np.array([calculate_local_time(t_UTC_time, lon) for lon in lon[0,:]])
        # Select areas where local time is target local time (e.g., 6 AM)
        t_selected_indices = np.where([(lt.year == year) & (lt.day == int(doy_to_yearyyyymmdd(year, doy)[7:8])) & (target_local_time - 1 <= lt.hour <= target_local_time + 1) for lt in t_local_times])[0]
    
        if t_selected_indices.size>0:
            t_var_LT_combined[:, t_selected_indices] = t_var[:, t_selected_indices]
    print(doy_to_yearyyyymmdd(year, doy), 'at ', target_local_time ,' local time.')

    return t_var_LT_combined

