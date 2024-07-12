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

def extract_filelist_doy_he5(directory, year):
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
                nc_files = glob.glob(os.path.join(sub_dir_path, '*.he5'))

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

# Function to replace inf/-inf with NaN in data
def replace_inf_with_nan(lat, lon, data):
    inf_mask = np.isinf(lat) | np.isinf(lon)
    data[inf_mask] = np.nan
    return data
