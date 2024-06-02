import numpy as np
import pandas as pd
import scipy.io
import os
import h5py
import netCDF4
from datetime import datetime, timedelta

def convert_to_local_time(df):
    # Function to convert fraction of days since 1900-01-01 00:00:00 UTC to local time
    # Calculate UTC time
    base_date = datetime(1900, 1, 1, 0, 0, 0)
    utc_time = base_date + pd.to_timedelta(df['time'], unit='D')

    # Calculate timezone offset based on longitude
    timezone_offset_hours = df['lon'] / 15  # 15 degrees per hour
    timezone_offset = pd.to_timedelta(timezone_offset_hours, unit='H')

    # Convert to local time
    local_time = utc_time + timezone_offset

    # Convert local time back to fraction of days since 1900-01-01 00:00:00
    local_time_fraction_days = (local_time - base_date) / timedelta(days=1)

    return local_time_fraction_days
    
def calculate_doy(dt, base_year):
    year_start = datetime(base_year, 1, 1)
    doy = (dt - year_start).days + 1
    return doy
    
def load_mat_file(mat_file, path):
    print('This function will be deprecated since HydroAI no longer depends on MATLAB files.')
    # Open MATLAB file
    with h5py.File(mat_file, 'r') as file:
        # Get all variable names
        var_names = list(file.keys())

        # Save variables with their names
        variables = {}
        for var_name in var_names:
            var_value = file[var_name][:]
            variables[var_name] = var_value

    # Transpose specific variables
    ASCAT_SM = variables['ASCAT_SM_'+path].transpose(2, 1, 0)
    latitude = variables['ascat_v_lat'].transpose(1, 0)
    longitude = variables['ascat_v_lon'].transpose(1, 0)

    return ASCAT_SM, latitude, longitude

def load_porosity_mat(mat_file):
    print('This function will be deprecated since HydroAI no longer depends on MATLAB files.')
    if os.path.exists(mat_file):
        data = scipy.io.loadmat(mat_file)
        variables = {}
    for var_name in data:
        variables[var_name] = data[var_name]
    return variables

def create_netcdf_file(nc_file, latitude, longitude, VAR, var_name='ASCAT_SM'):
    # Create a new NetCDF file
    nc_data = netCDF4.Dataset(nc_file, 'w')

    # Define the dimensions
    
    if VAR.ndim > 2:        
        rows, cols, doy = VAR.shape
    else:
        rows, cols = VAR.shape

    # Create dimensions in the NetCDF file
    nc_data.createDimension('latitude', rows)
    nc_data.createDimension('longitude', cols)
    
    if VAR.ndim > 2:        
        nc_data.createDimension('doy', doy)

    # Create latitude and longitude variables
    lat_var = nc_data.createVariable('latitude', 'f4', ('latitude', 'longitude'))
    lon_var = nc_data.createVariable('longitude', 'f4', ('latitude', 'longitude'))

    # Create variables for VAR
    if VAR.ndim > 2:        
        var = nc_data.createVariable(var_name, 'f4', ('latitude', 'longitude', 'doy'))
    else:
        var = nc_data.createVariable(var_name, 'f4', ('latitude', 'longitude'))
        
    # Assign data to the variables
    lat_var[:] = latitude
    lon_var[:] = longitude
    var[:]     = VAR

    # Close the NetCDF file
    nc_data.close()