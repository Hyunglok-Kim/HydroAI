import os
from netCDF4 import Dataset
import numpy as np

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
    nc_file_paths.sort
    
    return nc_file_paths

# Example usage:
#nc_paths = get_nc_file_paths(LIS_LSM_FP)
#print(f"Found {len(nc_paths)} .nc files.")

def get_nc_variable_names_units(nc_file_path):
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

def get_variable_from_nc(nc_file_path, variable_name, layer_index=0):
    """
    Extract a specific layer (if 3D) or the entire array (if 2D) of a variable 
    from a NetCDF file and return it as a NumPy array, with fill values replaced by np.nan.

    :param nc_file_path: Path to the NetCDF file.
    :param variable_name: Name of the variable to extract.
    :param layer_index: The index of the layer to extract if the variable is 3D. Default is 0.
    :return: NumPy array of the specified layer or the entire variable data, with np.nan for fill values.
    """
    with Dataset(nc_file_path, 'r') as nc:
        # Check if the variable exists in the NetCDF file
        if variable_name in nc.variables:
            variable = nc.variables[variable_name]
            
            # Set auto mask and scale to True to automatically convert fill values to NaN
            variable.set_auto_maskandscale(True)

            # Extract data based on the number of dimensions
            if variable.ndim == 3:
                # Extract the specified layer for 3D variables
                layer_data = variable[layer_index, :, :]
            elif variable.ndim == 2:
                # Extract all data for 2D variables
                layer_data = variable[:]
            else:
                raise ValueError(f"Variable '{variable_name}' is not 2D or 3D and is not supported by this function.")
            
            # Convert masked array to a regular array with np.nan for fill values
            return np.flipud(layer_data.filled(np.nan))
        else:
            raise ValueError(f"Variable '{variable_name}' does not exist in the NetCDF file.")
            
# Example usage:
#nc_file = nc_paths[0]
#nth_layer = 0  # Replace with the layer index you want to extract if variable is 3D
#variable_layer_data = get_variable_layer_from_nc(nc_file, 'lat',layer_index=nth_layer)
#print(f"Data for variable '{variable_name}':")
#print(variable_layer_data)