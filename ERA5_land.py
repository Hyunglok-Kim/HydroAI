"""
ERA5_land.py: A module for processing reanalysis ERA5-based soil moisture data.

(for monthly dataset)

"""
import os
import glob
import datetime
import numpy as np
import h5py
import netCDF4
from tqdm import tqdm
import calendar

def preprocess_lon_lat(lon, lat):
    # lat, lon edit for standard data shape
    lat = lat.reshape(-1, 1)
    lat = np.repeat(lat, len(lon), axis=1)
    lon -= 180
    lon = np.tile(lon, (len(lat), 1))
  
    return lon, lat

def correct_shape(var_data):
    # Transpose due to ERA5's raw shape is (12, 1801, 3600) = (month, lat, lon)
    var_data = np.transpose(var_data, (1,2,0))
    lon_size = np.shape(var_data)[1]
    var_data_0_180 = var_data[:,:lon_size//2,:]
    var_data_180_0 = var_data[:,lon_size//2:,:]
    var_data = np.concatenate((var_data_180_0, var_data_0_180), axis=1)

    return var_data

def create_mask(nc_data, var, var_data):
    try:
        # Replace Fill value into np.nan
        fill_value = np.float64(nc_data.variables[var]._FillValue)
        var_data[var_data == fill_value] = np.nan
    except AttributeError:
        print(f"{var} has no FillValue attribute!")
    return var_data
