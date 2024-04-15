import scipy.io
import os
import h5py
import netCDF4

def load_mat_file(mat_file, path):
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