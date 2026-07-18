"""
SWE.py: A module for processing AMSR .he5 snow water equivalent (SWE) data.

Shared file-listing logic lives in readers_common.
"""
import numpy as np

from . import readers_common


def extract_filelist_doy_he5(directory, year):
    """List .he5 files and their DOY from 'yyyy.mm.dd' subdirectories."""
    return readers_common.extract_filelist_doy(directory, year, ext='he5')


# Function to replace inf/-inf with NaN in data
def replace_inf_with_nan(lat, lon, data):
    inf_mask = np.isinf(lat) | np.isinf(lon)
    data[inf_mask] = np.nan
    return data
