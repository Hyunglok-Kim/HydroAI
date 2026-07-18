"""
SMOS_IC.py: A module for processing SMOS (Soil Moisture Ocean Salinity) satellite IC algorithm soil moisture data.

Shared file-listing/array-stacking logic lives in readers_common.
"""
from . import readers_common


def extract_filelist_doy(directory, year):
    """List .nc files and their DOY from 'yyyy.mm.dd' subdirectories."""
    return readers_common.extract_filelist_doy(directory, year, ext='nc')


def create_array_from_nc(file_list, data_doy, year, variable_name):
    """Stack a variable from daily .nc files into a (x, y, doy) array."""
    return readers_common.create_array_from_nc(file_list, data_doy, year, variable_name)
