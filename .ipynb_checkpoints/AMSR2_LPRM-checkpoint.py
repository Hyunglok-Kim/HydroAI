"""
SMAP.py: A module for processing GCOM-W1 satellite AMSR2 LPRM-based soil moisture data.

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
    Extracts a list of .nc files and their corresponding day of the year (DOY) from a directory.

    Args:
        directory (str): The directory containing .nc files organized in 'yyyy.mm.dd' subdirectories.
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
                # Search for .nc files within the subdirectory
                nc_files = glob.glob(os.path.join(sub_dir_path, '*.nc4'))

                # Get the day of the year (DOY) number
                doy = date_obj.timetuple().tm_yday

                # Append file paths and corresponding DOY to the data list
                for file in nc_files:
                    data.append((file, doy))

    # Sort the data list based on DOY (and date)
    data.sort(key=lambda x: x[1])

    # Unzip the sorted data into separate lists
    file_list, data_doy = zip(*data) if data else ([], [])

    return file_list, data_doy

def correct_shape(data):
    data = np.fliplr(data.transpose((1, 0, 2)))

    return data

# bit meaning
# 1 Negative optical depth in X
# 2 Negative optical depth in C2
# 3 Negative optical depth in C1
# 4 High optical depth value in X
# 5 High optical depth value in C2
# 6 High optical depth value in C1
# 7 No valid data
# 8 Ice
# 9 Not processed
def create_mask(bit_mask_3d, bit_position=8):
    # Ensure the bit mask is in an integer format
    bit_mask_3d = bit_mask_3d.astype(np.int32)
    
    # Identify where bit 8 (Ice) is set
    ice_mask = (bit_mask_3d & (1 << bit_position)) != 0

    # Convert boolean mask to integer mask (1 for ice, 0 for no ice)
    ice_mask = ice_mask.astype(int)

    return ice_mask

def create_array_from_nc(file_list, data_doy, year, variable_name):
    # Read data from the first NC file
    nc_data = netCDF4.Dataset(file_list[0])
    t_data = np.flipud(nc_data.variables[variable_name][:])
    x, y = t_data.shape[:2]

    doy_max = 366 if calendar.isleap(year) else 365

    # Create the array filled with NaN
    data_array = np.empty((x, y, doy_max + 1))
    data_array[:] = np.nan

    # Loop over the file list with a progress bar
    for i, nc_file in enumerate(tqdm(file_list, desc="Processing files")):
        # Read data from the NC file
        nc_data = netCDF4.Dataset(nc_file)
        t_data = np.flipud(nc_data.variables[variable_name][:])
    
        # Get the corresponding doy value from the data_doy list
        doy = data_doy[i]

        # Assign the data to the array
        data_array[:, :, doy] = t_data

        # Close the NC file
        nc_data.close()

    return data_array
