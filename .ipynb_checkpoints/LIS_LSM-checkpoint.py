import os
from netCDF4 import Dataset
import numpy as np
import datetime

def get_nc_file_paths(base_dir, contain='_HIST_'):
    """
    Get a list of file paths to all .nc files within the base_dir directory,
    excluding files that contain '_HIST_' in the filename.

    :param base_dir: The base directory to search for .nc files.
    :return: A list of file paths to .nc files.
    """
    nc_file_paths = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Check for .nc files not containing '_HIST_'
            if file.endswith('.nc') and contain in file:
                # Construct the full file path
                full_path = os.path.join(root, file)
                nc_file_paths.append(full_path)

    # Sort the file paths in descending order
    nc_file_paths.sort()
    
    return nc_file_paths

# Example usage:
#nc_paths = get_nc_file_paths(LIS_LSM_FP)
#print(f"Found {len(nc_paths)} .nc files.")

def get_nc_variable_names_units(nc_file_path):
    print('This function will be deprecated. Use HydroAI Dataset.get_nc_variable_names_units instead')
    """
    Get a list of variable names and a corresponding list of their units from a NetCDF file.

    :param nc_file_path: Path to the NetCDF file.
    :return: A list of variable names and a list of units for these variables.
    """
    variable_names = []
    variable_units_list = []
    
    with Dataset(nc_file_path, 'r') as nc:
        # Extract the list of variable names
        variable_names = list(nc.variables.keys())
        
        # Create a list that contains only the units for each variable
        for var_name in variable_names:
            try:
                # Try to get the 'units' attribute for the variable
                variable_units_list.append(nc.variables[var_name].units)
            except AttributeError:
                # If the variable doesn't have a 'units' attribute, append None
                variable_units_list.append(None)
                
    return variable_names, variable_units_list


# Example usage:
#nc_file = nc_paths[0]
#variables = get_nc_variable_list(nc_file)
#print("Variables in the NC file:", variables)

def get_variable_from_nc(nc_file_path, variable_name, layer_index=0, flip_data=True):
    print('This function will be deprecated. Use HydroAI Dataset.get_variable_from_nc instead')
    """
    Extract a specific layer (if 3D), the entire array (if 2D or 1D), or the value (if 0D) of a variable
    from a NetCDF file and return it as a NumPy array, with fill values replaced by np.nan.

    :param nc_file_path: Path to the NetCDF file.
    :param variable_name: Name of the variable to extract.
    :param layer_index: The index of the layer to extract if the variable is 3D. Default is 0.
    :return: NumPy array or scalar of the specified variable data, with np.nan for fill values.
    """
    with Dataset(nc_file_path, 'r') as nc:
        # Check if the variable exists in the NetCDF file
        if variable_name in nc.variables:
            variable = nc.variables[variable_name]
            
            # Set auto mask and scale to True to automatically convert fill values to NaN
            variable.set_auto_maskandscale(True)

            # Extract data based on the number of dimensions
            if variable.ndim == 4:
                # Extract the specified layer for 3D variables
                #it is very likely some ERA5 data
                if layer_index == 'all':
                    data = variable[:,0,:,:]
                else:
                    data = variable[layer_index,0,:,:]
            
            elif variable.ndim == 3:
                # Extract the specified layer for 3D variables
                if layer_index == 'all':
                    data = variable
                else:
                    data = variable[layer_index,:,:]
                
            elif variable.ndim == 2:
                # Extract all data for 2D variables
                data = variable[:, :]
            elif variable.ndim == 1:
                # Extract all data for 1D variables
                data = variable[:]
            elif variable.ndim == 0:
                # Extract scalar value for 0D variables
                data = variable.scalar()
            else:
                raise ValueError(f"Variable '{variable_name}' has unsupported number of dimensions: {variable.ndim}.")

            # Handle fill values (mask to NaN if necessary)
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(np.nan)
            
            # Flip the data upside down if it's 2D or 3D (common in geographical data to match orientation)
            if flip_data:
                if variable.ndim in [2, 3]:
                    data = np.flipud(data)

            return data
        else:
            raise ValueError(f"Variable '{variable_name}' does not exist in the NetCDF file.")

# Example usage:
#nc_file = nc_paths[0]
#nth_layer = 0  # Replace with the layer index you want to extract if variable is 3D
#variable_layer_data = get_variable_layer_from_nc(nc_file, 'lat',layer_index=nth_layer)
#print(f"Data for variable '{variable_name}':")
#print(variable_layer_data)

def parse_date_from_path(file_path):
    """
    Extracts the datetime from the file path based on a known pattern.
    Assumes file names contain dates in the format 'yyyymmddhhmm' directly before '.d01.nc'.
    """
    date_str = file_path.split('_')[-1][:-7]  # Get the date part just before '.d01.nc'
    return datetime.datetime.strptime(date_str, '%Y%m%d%H%M')

def find_date_index(nc_file_paths, target_date_str):
    """
    Finds the index of the file in nc_file_paths whose date is closest to the target_date_str.
    
    :param nc_file_paths: List of file paths containing date strings.
    :param target_date_str: Target date string in 'yyyymmddhhmmss' format.
    :return: Index of the file closest to the specified date.
    """
    target_date = datetime.datetime.strptime(target_date_str, '%Y%m%d%H%M%S')

    # Initialize variables to keep track of the minimum difference and the index
    min_diff = float('inf')
    closest_index = -1

    # Loop through each file, parse the date, calculate difference, and update the closest index
    for index, file_path in enumerate(nc_file_paths):
        file_date = parse_date_from_path(file_path)
        diff = abs((target_date - file_date).total_seconds())  # Difference in seconds

        if diff < min_diff:
            min_diff = diff
            closest_index = index

    return closest_index