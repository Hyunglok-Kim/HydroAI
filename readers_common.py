"""
readers_common.py: Shared helpers for the gridded satellite/model data readers.

The mission-specific reader modules (SMAP, AMSR2_LPRM, SMOS_IC, SWE, ...) keep
their public function names but delegate the common logic to this module.
"""
import os
import glob
import datetime
import calendar
import numpy as np
import netCDF4
from tqdm import tqdm


def extract_filelist_doy(directory, year, ext='nc'):
    """
    Extract data files and their day of year (DOY) from 'yyyy.mm.dd' subdirectories.

    Args:
        directory (str): Directory containing 'yyyy.mm.dd' subdirectories.
        year (int): Year to extract.
        ext (str): File extension to match without the dot (e.g. 'nc', 'nc4', 'h5', 'he5').

    Returns:
        tuple: (file_list, data_doy) sorted by DOY.
    """
    data = []

    for subdir in os.listdir(directory):
        sub_dir_path = os.path.join(directory, subdir)
        # Only 'yyyy.mm.dd' directories
        if os.path.isdir(sub_dir_path) and len(subdir) == 10 and subdir[4] == '.' and subdir[7] == '.':
            date_str = '-'.join(subdir.split('.'))
            date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            if date_obj.year == year:
                files = glob.glob(os.path.join(sub_dir_path, f'*.{ext}'))
                doy = date_obj.timetuple().tm_yday
                for file in files:
                    data.append((file, doy))

    data.sort(key=lambda x: x[1])
    file_list, data_doy = zip(*data) if data else ([], [])

    return file_list, data_doy


def create_array_from_nc(file_list, data_doy, year, variable_name):
    """
    Stack a variable from daily NetCDF files into a (x, y, doy) array.

    Index 0 of the DOY axis is unused so that data_array[:, :, doy] can be
    indexed with 1-based DOY values.
    """
    nc_data = netCDF4.Dataset(file_list[0])
    t_data = np.flipud(nc_data.variables[variable_name][:])
    x, y = t_data.shape[:2]
    nc_data.close()

    doy_max = 366 if calendar.isleap(year) else 365
    data_array = np.full((x, y, doy_max + 1), np.nan)

    for i, nc_file in enumerate(tqdm(file_list, desc="Processing files")):
        nc_data = netCDF4.Dataset(nc_file)
        t_data = np.flipud(nc_data.variables[variable_name][:])
        data_array[:, :, data_doy[i]] = t_data
        nc_data.close()

    return data_array
