"""
SMAP.py: A module for processing SMAP (Soil Moisture Active Passive) satellite data.

This module contains functions for extracting file lists based on the day of the year (DOY), 
generating latitude and longitude grids, creating arrays from .h5 files, and creating NetCDF files 
from the processed data.
"""

import os
import glob
import datetime
import numpy as np
import h5py
import netCDF4
from tqdm import tqdm
import calendar

def extract_filelist_doy(directory, year):
    """
    Extracts a list of .h5 files and their corresponding day of the year (DOY) from a directory.

    Args:
        directory (str): The directory containing .h5 files organized in 'yyyy.mm.dd' subdirectories.
        year (int): The year for which the files are to be extracted.

    Returns:
        tuple: Two lists, one of file paths and one of corresponding DOYs.
    """
    data = []

    # Iterate over the subdirectories within the specified directory
    for subdir in os.listdir(directory):
        sub_dir_path = os.path.join(directory, subdir)
        
        # Check if the item is a directory and matches the 'yyyy.mm.dd' format
        if os.path.isdir(sub_dir_path) and len(subdir) == 10 and subdir[4] == '.' and subdir[7] == '.':
            # Convert the 'yyyy.mm.dd' format to 'yyyy-mm-dd'
            date_str = '-'.join(subdir.split('.'))
            # Convert the date string to a datetime object
            date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')

            # Process only if the year matches the specified year
            if date_obj.year == year:
                # Search for .h5 files within the subdirectory
                h5_files = glob.glob(os.path.join(sub_dir_path, '*.h5'))

                # Get the day of the year (DOY) number
                doy = date_obj.timetuple().tm_yday

                # Append file paths and corresponding DOY to the data list
                for file in h5_files:
                    data.append((file, doy))

    # Sort the data list based on DOY (and date)
    data.sort(key=lambda x: x[1])

    # Unzip the sorted data into separate lists
    file_list, data_doy = zip(*data) if data else ([], [])

    return file_list, data_doy

def get_e2grid(cpuserver_data_FP, mission_product):
    """
    Gets the EASE2 grid longitude and latitude for the specified mission product.

    Args:
        cpuserver_data_FP (str): File path where grid files are located.
        mission_product (str): The product name, used to determine grid resolution.

    Returns:
        tuple: Two arrays containing the longitude and latitude values of the grid.
    """

    if mission_product.startswith('SPL3SMP.'):
        grid_prefix = 'EASE2_M36km'
        shape = (964, 406)
    elif mission_product.startswith('SPL3SMP_E.'):
        grid_prefix = 'EASE2_M09km'
        shape = (3856, 1624)
    elif mission_product.startswith('25km'):
        grid_prefix = 'EASE2_M25km'
        shape = (1388, 584)
    elif mission_product.startswith('3km'):
        grid_prefix = 'EASE2_M03km'
        shape = (11568, 4872)
    else:
        return None, None  # or some default value, or raise an error

    # Construct file paths for latitude and longitude
    # File names use shape in the order (lat_size, lon_size)
    grid_size = f"{shape[0]}x{shape[1]}x1"
    lats_FP = os.path.join(cpuserver_data_FP, f'grids/{grid_prefix}.lats.{grid_size}.double')
    lons_FP = os.path.join(cpuserver_data_FP, f'grids/{grid_prefix}.lons.{grid_size}.double')

    # Load and reshape data
    # Reshape uses flipped dimensions (lon_size, lat_size)
    latitude = np.fromfile(lats_FP, dtype=np.float64).reshape((shape[1], shape[0]))
    longitude = np.fromfile(lons_FP, dtype=np.float64).reshape((shape[1], shape[0]))

    return longitude, latitude

def create_array_from_h5(file_list, data_doy, year, cpuserver_data_FP, mission_product, variable_name, group_name):
    """
    Creates a 3D numpy array from a list of .h5 files containing variable data for each DOY.

    Args:
        file_list (list): List of .h5 file paths.
        data_doy (list): List of corresponding DOYs for each file.
        year (int): The year for which data is processed.
        cpuserver_data_FP (str): File path where grid files are located.
        mission_product (str): The mission product name.
        variable_name (str): The variable name within the .h5 files.
        group_name (str): The group name within the .h5 files.

    Returns:
        tuple: A 3D array of data, and 2D arrays of longitude and latitude.
    """
    x, y = None, None
    doy_max = 366 if calendar.isleap(year) else 365

    # Initialize data_array with NaNs
    data_array = None

    # Loop over the file list with a progress bar
    for i, h5_file in enumerate(tqdm(file_list, desc="Processing files", unit="file")):
        try:
            with h5py.File(h5_file, 'r') as hdf5_data:
                dataset = hdf5_data[group_name][variable_name]
                
                if data_array is None:
                    # Get the shape of dataset from the first file
                    x, y = dataset[:].shape
                    data_array = np.full((x, y, doy_max + 1), np.nan)  # Create the array filled with NaN

                t_data = dataset[:].astype(np.float64)

                # Get attributes and apply them if they exist
                fill_value = np.float64(dataset.attrs.get('_FillValue', np.nan))
                valid_min = np.float64(dataset.attrs.get('valid_min', -np.inf))
                valid_max = np.float64(dataset.attrs.get('valid_max', np.inf))
                scale_factor = np.float64(dataset.attrs.get('scale_factor', 1.0))
                add_offset = np.float64(dataset.attrs.get('add_offset', 0.0))

                # Apply scale and offset if applicable
                if 'scale_factor' in dataset.attrs or 'add_offset' in dataset.attrs:
                    t_data = t_data * scale_factor + add_offset

                # Mask invalid data
                t_data = np.where((t_data < valid_min) | (t_data > valid_max) | (t_data == fill_value), np.nan, t_data)

                # Get the corresponding doy value from the data_doy list
                doy = data_doy[i]

                # Assign the data to the array
                data_array[:, :, doy] = t_data

        except OSError as e:
            print(f"Error processing file {h5_file}: {e}")
            if data_array is not None and x is not None and y is not None:
                # Mark the entire day as NaN in case of an error
                data_array[:, :, data_doy[i]] = np.nan

    # Get EASE2 lat/lon from the get_e2grid function
    longitude, latitude = get_e2grid(cpuserver_data_FP, mission_product)
    
    return data_array, longitude, latitude

def create_netcdf_file(nc_file, longitude, latitude, **data_vars):
    """
    Creates a NetCDF file from the provided data arrays and latitude/longitude grids.

    Args:
        nc_file (str): Path to the output NetCDF file.
        latitude (np.array): 2D array of latitude values.
        longitude (np.array): 2D array of longitude values.
        data_vars (dict): Dictionary of 3D data arrays to include in the NetCDF file.

    Returns:
        None
    """
    # Create a new NetCDF file
    nc_data = netCDF4.Dataset(nc_file, 'w')

    # Define the dimensions
    rows, cols = latitude.shape
    # Assuming all data variables have the same 'time' dimension size
    doy = next(iter(data_vars.values())).shape[2]

    # Create dimensions in the NetCDF file
    nc_data.createDimension('latitude', rows)
    nc_data.createDimension('longitude', cols)
    nc_data.createDimension('doy', doy)

    # Create latitude and longitude variables
    lat_var = nc_data.createVariable('latitude', 'f4', ('latitude', 'longitude'))
    lon_var = nc_data.createVariable('longitude', 'f4', ('latitude', 'longitude'))

    # Assign data to the latitude and longitude variables
    lat_var[:] = latitude
    lon_var[:] = longitude

    # Create variables and assign data for each item in data_vars
    for var_name, var_data in data_vars.items():
        # Create variable in NetCDF file
        nc_var = nc_data.createVariable(var_name, 'f4', ('latitude', 'longitude', 'doy'))
        # Assign data to the variable
        nc_var[:] = var_data

    # Close the NetCDF file
    nc_data.close()

    print(f"NetCDF file {nc_file} created successfully.")

