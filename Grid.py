import numpy as np
from ease_lonlat import EASE2GRID

### EASE2 grid generator ###
def get_e2_grid(resolution_key):
    e2_grid_params = {
        '1km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83, 'res': 1000.9, 'n_cols': 34704, 'n_rows': 14616},
        '3km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83, 'res': 3002.69, 'n_cols': 11568, 'n_rows': 4872},
        '3.125km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7307375.92, 'res': 3128.16, 'n_cols': 11104, 'n_rows': 4672},
        '6.25km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7307375.92, 'res': 6256.32, 'n_cols': 5552, 'n_rows': 2336},
        '9km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83, 'res': 9008.05, 'n_cols': 3856, 'n_rows': 1624},
        '12.5km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7307375.92, 'res': 12512.63, 'n_cols': 2776, 'n_rows': 1168},
        '25km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7307375.92, 'res': 25025.26, 'n_cols': 1388, 'n_rows': 584},
        '36km': {'epsg': 6933, 'x_min': -17367530.44, 'y_max': 7314540.83, 'res': 36032.22, 'n_cols': 964, 'n_rows': 406}
    }
    # Prefer SUPPORTED_GRIDS if available
    grid_params = e2_grid_params[resolution_key]

    # Initialize the EASE2GRID with the specified parameters
    grid = EASE2GRID(
        name=f'EASE2_G{resolution_key.replace("km", "")}',
        epsg=grid_params['epsg'],
        x_min=grid_params['x_min'],
        y_max=grid_params['y_max'],
        res=grid_params['res'],
        n_cols=grid_params['n_cols'],
        n_rows=grid_params['n_rows']
    )
    return grid

def generate_lat_lon_e2grid(resolution_key):
    # Initialize the grid using the previous function
    grid = get_e2_grid(resolution_key)

    # Determine the number of columns and rows from the grid
    n_cols = grid.n_cols
    n_rows = grid.n_rows

    # Create empty arrays to store latitude and longitude
    latitudes = np.zeros((n_rows, n_cols))
    longitudes = np.zeros((n_rows, n_cols))

    # Iterate over each cell in the grid
    for row in range(n_rows):
        for col in range(n_cols):
            lon, lat = grid.rc2lonlat(col, row)
            longitudes[row, col] = lon
            latitudes[row, col] = lat

        # Optionally print progress
        if row % 100 == 0:
            print(f"Processing row {row}/{n_rows}")

    return longitudes, latitudes