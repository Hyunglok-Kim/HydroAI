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
import os
import glob

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

def mode_function(x):
    return x.mode().iloc[0]

def Resampling(lat_target, lon_target, lat_input, lon_input, VAR, sampling_method, agg_method='mean', mag_factor=3):
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
    
    def magnify_VAR(lat_input, lon_input, VAR, mag_factor):
        # Rescale lon and lat using bilinear interpolation (order=1)
        m_lon = zoom(lon_input, mag_factor, order=1)
        m_lat = zoom(lat_input, mag_factor, order=1)
        m_values = zoom(VAR, mag_factor, order=0)  # Nearest neighbor interpolation
        return m_lat, m_lon, m_values
    
    # Check if magnification is needed
    if lat_target.shape > lat_input.shape:
        lat_input, lon_input, VAR = magnify_VAR(lat_input, lon_input, VAR, mag_factor)

    # Flatten the target and input coordinates
    target_coords = np.column_stack([lat_target.ravel(), lon_target.ravel()])
    input_coords = np.column_stack([lat_input.ravel(), lon_input.ravel()])

    # Create a KDTree for fast nearest-neighbor lookup
    tree = cKDTree(input_coords)

    # Find the nearest neighbors in the input data for each target location
    distances, indices = tree.query(target_coords, k=1)  # k=1 for nearest neighbor

    # Using the indices to map input data to target data grid
    resampled_VAR = VAR.ravel()[indices].reshape(lat_target.shape)

    # Handling aggregation if necessary
    if agg_method != 'mean':  # 'mean' is handled by default above
        # Example: Handling other aggregation methods
        df = pd.DataFrame({'values': resampled_VAR.ravel(), 'indices': indices})
        if agg_method == 'max':
            resampled_VAR = df.groupby('indices')['values'].max().values.reshape(lat_target.shape)
        elif agg_method == 'min':
            resampled_VAR = df.groupby('indices')['values'].min().values.reshape(lat_target.shape)
        # Implement other aggregations as needed

    return resampled_VAR
    
def Resampling_old(lat_target, lon_target, lat_input, lon_input, VAR, sampling_method, agg_method='mean', mag_factor=3):
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
    #-----------------------------------------------------------------%
    def magnify_VAR(lat_input, lon_input, VAR, mag_factor):
        # Rescale lon and lat using bilinear interpolation (order=1)
        m_lon = zoom(lon_input, [mag_factor, mag_factor], order=1)
        m_lat = zoom(lat_input, [mag_factor, mag_factor], order=1) 
        # order=0, the zoom function performs nearest-neighbor interpolation
        m_values = zoom(VAR, [mag_factor, mag_factor], order=0)
        return m_lat, m_lon, m_values
    
    s_target = lat_target.shape
    s_input  = lat_input.shape
    
    if s_target[0] > s_input[0] or s_target[1] > s_input[1]:
        lat_input, lon_input, VAR = magnify_VAR(lat_input, lon_input, VAR, mag_factor)

    def resample_agg(lat_target, lon_target, lat_input, lon_input, VAR, sampling_method, agg_method):
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
        
        if agg_method == 'mode':
            agg_method = mode_function
            agg_values = df.groupby('idx')['val'].apply(agg_method)
        else:
            agg_values = getattr(df.groupby('idx')['val'], agg_method)()
            
        VAR_r[np.unravel_index(agg_values.index.values, VAR_r.shape)] = agg_values.values

        return VAR_r

    if lat_target.shape == lat_input.shape and np.all(lat_target == lat_input) and np.all(lon_target == lon_input):
        print('Resampling is not required.')
        VAR_r = VAR
    else:
        VAR_r = resample_agg(lat_target, lon_target, lat_input, lon_input, VAR, sampling_method, agg_method)
    return VAR_r

def process_var(i, lat_target, lon_target, lat_input, lon_input, data, sampling_method,agg_method, mag_factor):
    #print(i)
    VAR = data[:,:,i]
    result = Resampling(lat_target, lon_target, lat_input, lon_input, VAR, sampling_method, agg_method, mag_factor)
    return result

def Resampling_forloop(lat_target, lon_target, lat_input, lon_input, VAR, sampling_method='nearest', agg_method='mean', mag_factor=3):
    
    m, n = lat_target.shape  # Get the dimensions from lat_target
    # Initialize results array
    results = np.empty((m, n, VAR.shape[2]))
    
    for i in tqdm(range(0, VAR.shape[2])):
        t = Resampling(lat_target, lon_target, lat_input, lon_input, VAR[:,:,i],'nearest')
        results[:,:,i] = t

    return results

def Resampling_parallel(lat_target, lon_target, lat_input, lon_input, VAR, sampling_method='nearest',agg_method='mean', mag_factor=3):

    # Create a partial function with the arguments that don't change
    partial_process_var = partial(process_var, lat_target=lat_target, lon_target=lon_target,
                                  lat_input=lat_input, lon_input=lon_input, data=VAR, sampling_method=sampling_method, agg_method=agg_method, mag_factor=mag_factor)
    m, n = lat_target.shape  # Get the dimensions from lat_target

    # Initialize results array
    results = np.empty((m, n, VAR.shape[2]))

    with Pool(8) as p:
        results_list = p.map(partial_process_var, range(VAR.shape[2]))

    for i, result in enumerate(results_list):
        results[:,:,i] = result

    return results

def moving_average_3d(data, window_size):
    m, n, z = data.shape
    padding = window_size // 2  # Number of elements to pad on each side
    
    # Pad the data with NaN values to handle the edges
    padded_data = np.pad(data, ((0, 0), (0, 0), (padding, padding)), mode='constant', constant_values=np.nan)
    
    # Create an array to store the moving averaged values
    moving_averaged = np.zeros((m, n, z))
    
    # Calculate the moving average for each row and pixel
    # This code can be modified not to average already "moving averaged" values
    # This code can be modified not to average if there is less than certain numbers of valid points
    for k in range(z):
        moving_averaged[:, :, k] = np.nanmean(padded_data[:, :, k:k+window_size], axis=2)
    
    return moving_averaged

def find_closest_index(longitudes, latitudes, point):
    lon_lat = np.c_[longitudes.ravel(), latitudes.ravel()]
    tree = cKDTree(lon_lat)
    dist, idx = tree.query(point, k=1)
    return np.unravel_index(idx, latitudes.shape)

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
        pattern = f"{directory_path}/**/*{file_extension}"
    else:
        pattern = f"{directory_path}/*{file_extension}"

    # Get a list of all files matching the pattern
    file_paths = glob.glob(pattern, recursive=recursive)

    # Filter files to include only those containing any of the filter_strs, if provided
    if filter_strs:
        filtered_paths = []
        for file_path in file_paths:
            if any(substring in os.path.basename(file_path) for substring in filter_strs):
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
# filtered_txt_files = get_file_list(directory, file_ext, filter_strs=["abs", "2021", "report"])

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