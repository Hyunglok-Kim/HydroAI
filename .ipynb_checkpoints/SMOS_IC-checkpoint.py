import netCDF4
import numpy as np
import calendar
import os
import tarfile
import glob
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature

def extract_tgz_files(root_dir, year):
    """
    Extract all .tgz files found in subdirectories of the specified root directory.

    Parameters:
    root_dir (str): Path to the root directory.

    Returns:
    None
    """
    # Iterate over all subdirectories of root_dir/A/year and root_dir/D/year
    for subdir in ['A', 'D']:
        dir_path = os.path.join(root_dir, subdir, str(year))
        for day_dir in os.listdir(dir_path):
            day_path = os.path.join(dir_path, day_dir)
            if os.path.isdir(day_path):
                # Find all .tgz files in the directory and extract them
                for file_name in os.listdir(day_path):
                    if file_name.endswith('.tgz'):
                        file_path = os.path.join(day_path, file_name)
                        # Check if the extracted files already exist
                        tar = tarfile.open(file_path, 'r:gz')
                        extract_dir = os.path.splitext(os.path.join(day_path, file_name))[0]
                        if not os.path.isdir(extract_dir):
                            # Extract the archive file if the extracted files don't already exist
                            tar.extractall(day_path)
                            tar.close() 

def extract_filelist_doy(directory):
    data_doy = []
    file_list = []

    # Iterate over the subdirectories within the specified directory
    for subdir in os.listdir(directory):
        sub_dir_path = os.path.join(directory, subdir)
        
        # Check if the item is a directory
        if os.path.isdir(sub_dir_path):
            # Search for NC files within the subdirectory
            nc_files = glob.glob(os.path.join(sub_dir_path, '*.nc'))
            
            # Add the found files to the file_list
            file_list.extend(nc_files)
            
            # Extract the day of the year (DOY) from the file names
            for nc_file in nc_files:
                file_name = os.path.basename(nc_file)
                # Extract the date string from the file name
                date_str = file_name.split('_')[4][:8]
                # Convert the date string to a datetime object
                date_obj = datetime.datetime.strptime(date_str, '%Y%m%d')
                # Get the day of the year (DOY) number
                doy = date_obj.timetuple().tm_yday
                # Add the DOY number to the data_doy list
                data_doy.append(doy)
    
    return file_list, data_doy

def apply_scale_offset(data, add_offset, scale_factor):
    return data * scale_factor + add_offset

def replace_fill_value(data, fill_value):
    data[data == fill_value] = np.nan
    return data

def create_array_from_nc(file_list, data_doy, year, variable_name):
    # Read data from the first NC file
    nc_data = netCDF4.Dataset(file_list[0])
    fill_value = np.float64(nc_data.variables[variable_name]._FillValue)
    #print(nc_data.variables)
    t_data        = np.flipud(nc_data.variables[variable_name][:])
    x, y = t_data.shape[:2]

    doy_max = 366 if calendar.isleap(year) else 365

    # Create the array filled with NaN
    data_array = np.empty((x, y, doy_max + 1))
    data_array[:] = np.nan

    # Loop over the file list
    for i, nc_file in enumerate(file_list):
        # Read data from the NC file
        nc_data = netCDF4.Dataset(nc_file)
        t_data = np.flipud(nc_data.variables[variable_name][:])
    
        # Get the corresponding doy value from the data_doy list
        doy = data_doy[i]

        # Assign the data to the array
        data_array[:, :, doy] = t_data

        # Close the NC file
        nc_data.close()
    
    nc_data = netCDF4.Dataset(file_list[0])
    lat = np.flipud(nc_data.variables['lat'][:])
    lon = (nc_data.variables['lon'][:])
    longitude, latitude = np.meshgrid(lon, lat)
   
    data_array = replace_fill_value(data_array, fill_value)
    
    if variable_name == 'RFI_Prob':
        add_offset    = nc_data.variables[variable_name].add_offset
        scale_factor  = nc_data.variables[variable_name].scale_factor
        data_array = apply_scale_offset(data_array, add_offset, scale_factor)

    return data_array, longitude, latitude

def create_netcdf_file(nc_file, latitude, longitude, SMOS_SM, SMOS_RFI, SMOS_SF=None):
    # Create a new NetCDF file
    nc_data = netCDF4.Dataset(nc_file, 'w')

    # Define the dimensions
    rows, cols, doy = SMOS_SM.shape

    # Create dimensions in the NetCDF file
    nc_data.createDimension('latitude', rows)
    nc_data.createDimension('longitude', cols)
    nc_data.createDimension('doy', doy)

    # Create latitude and longitude variables
    lat_var = nc_data.createVariable('latitude', 'f4', ('latitude', 'longitude'))
    lon_var = nc_data.createVariable('longitude', 'f4', ('latitude', 'longitude'))

    # Create variables for SMOS_SM, SMOS_RFI, and SMOS_SF
    smos_sm_var = nc_data.createVariable('SMOS_SM', 'f4', ('latitude', 'longitude', 'doy'))
    smos_rfi_var = nc_data.createVariable('SMOS_RFI', 'f4', ('latitude', 'longitude', 'doy'))

    if SMOS_SF is not None:
        smos_sf_var = nc_data.createVariable('SMOS_SF', 'f4', ('latitude', 'longitude', 'doy'))

    # Assign data to the variables
    lat_var[:] = latitude
    lon_var[:] = longitude
    smos_sm_var[:] = SMOS_SM
    smos_rfi_var[:] = SMOS_RFI

    if SMOS_SF is not None:
        smos_sf_var[:] = SMOS_SF

    # Close the NetCDF file
    nc_data.close()