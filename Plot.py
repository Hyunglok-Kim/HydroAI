import os
import re

from scipy import ndimage
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
from tqdm import tqdm
from PIL import Image
import rioxarray

from HydroAI.LIS_LSM import get_variable_from_nc
from HydroAI.Data import Resampling
import HydroAI.Data as Data

def plot_map(longitude, latitude, values, title, cmin, cmax, cmap='jet', bounds=None, dem_path=None):
    """
    Plots a map with the given data, either globally or within specified longitude and latitude bounds.

    Args:
    - longitude: 2D array of longitude values.
    - latitude: 2D array of latitude values.
    - values: 2D array of data values to plot.
    - title: Title for the colorbar and plot.
    - cmin, cmax: Minimum and maximum values for the colorbar.
    - cmap: Colormap to use for plotting data.
    - bounds: List or tuple of the format [lon_min, lon_max, lat_min, lat_max] for the map extent. If None, uses full range.
    - dem_path: Path to the DEM file for background in the plot (optional).
    
    Returns:
    - fig, ax: Figure and axes objects of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=200)
    
    # Set map extent if bounds are provided, else plot the entire range
    if bounds and len(bounds) == 4:
        lon_min, lon_max, lat_min, lat_max = bounds
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Plot DEM as background if provided
    if dem_path:
        with rasterio.open(dem_path) as dem:
            dem_data = dem.read(1)  # Read the first band
            dem_extent = [dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top]
            ax.imshow(dem_data, origin='upper', extent=dem_extent, transform=ccrs.PlateCarree(), cmap='terrain', alpha=0.5)

    im = ax.pcolormesh(longitude, latitude, values, transform=ccrs.PlateCarree(), cmap=cmap, vmin=cmin, vmax=cmax)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', edgecolor='black')
    ax.gridlines(color='gray', linestyle='--', linewidth=0.5)

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.5)
    cbar.set_label(title)
    im.set_clim(cmin, cmax)
    plt.show()
    
    return fig, ax
    
def plot_global_map(longitude, latitude, values, title, cmin, cmax, cmap='jet'):
    # Create a new figure and axes with a Plate Carrée projection
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=200)

    # Plot the values on the map
    im = ax.pcolormesh(longitude, latitude, values, transform=ccrs.PlateCarree(), cmap=cmap)

    # Add coastlines, ocean color, and national borders to the map
    ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', edgecolor='black')

    # Add gridlines to the map
    ax.gridlines(color='gray', linestyle='--', linewidth=0.5)
    
    # Add latitude and longitude labels
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cartopy.mpl.ticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cartopy.mpl.ticker.LatitudeFormatter())

    # Move the colorbar to the bottom and adjust its length
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.5)
    cbar.set_label(title)
    
    # Set the minimum and maximum values for the colorbar
    im.set_clim(cmin, cmax)

    # Show the plot
    plt.show()

def plot_regional_map(longitude, latitude, values, title, cmin, cmax, padding, cmap='jet', dem_path=None):
    lon_min, lon_max = np.min(longitude) - padding, np.max(longitude) + padding
    lat_min, lat_max = np.min(latitude) - padding, np.max(latitude) + padding

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=200)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Plot DEM as background if provided
    if dem_path:
        with rasterio.open(dem_path) as dem:
            dem_data = dem.read(1)  # Read the first band
            dem_extent = [dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top]
            ax.imshow(dem_data, origin='upper', extent=dem_extent, transform=ccrs.PlateCarree(), cmap='terrain', alpha=0.5)

    im = ax.pcolormesh(longitude, latitude, values, transform=ccrs.PlateCarree(), cmap=cmap, vmin=cmin, vmax=cmax)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', edgecolor='black')
    ax.gridlines(color='gray', linestyle='--', linewidth=0.5)

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.5)
    cbar.set_label(title)

    #plt.close(fig)  # Close the figure to free memory after saving
    return fig, ax

def plot_LULC_map_copernicus(longitude, latitude, rds, title, region=None):
    """
    Plot the LULC map from Copernicus data.
    
    Args:
    - longitude: Array of longitude values.
    - latitude: Array of latitude values.
    - rds: Raster dataset containing LULC data.
    - title: Title of the plot.
    - region (optional): If provided, should be a list or tuple in the format [lon_min, lon_max, lat_min, lat_max] 
                         specifying the bounding coordinates for the plot. If not provided, the function will use 
                         the full range of longitude and latitude values from the provided arrays.

    This function plots the land use and land cover (LULC) data, mapping the LULC codes to their respective colors 
    and displaying the result on a map. The map can be focused on a specific region if the 'region' parameter is provided.
    """
    
    # Define color map
    color_map = {
        0: (40, 40, 40),
        111: (88, 72, 31),
        112: (0, 153, 0),
        113: (112, 102, 62),
        114: (0, 204, 0),
        115: (78, 117, 31),
        116: (0, 120, 0),
        121: (102, 96, 0),
        122: (141, 180, 0),
        123: (141, 116, 0),
        124: (160, 220, 0),
        125: (146, 153, 0),
        126: (100, 140, 0),
        20: (255, 187, 34),
        30: (255, 255, 76),
        40: (240, 150, 255),  
        50: (250, 0, 0),
        60: (180, 180, 180),
        70: (240, 240, 240),
        80: (0, 50, 200),
        90: (0, 150, 160),
        100: (250, 230, 160),
        200: (200, 230, 255)
        # Add more LULC codes and their corresponding colors as needed
    }
    
    # Create an empty RGB array
    rgb_image = np.zeros((rds.shape[1], rds.shape[2], 3), dtype=np.uint8)
    
    # Map LULC values to RGB colors
    for code, color in color_map.items():
        mask = rds.values[0] == code
        rgb_image[mask] = color
    
    # Define land cover class names
    land_cover_classes = {
        0: 'No input data available',
        111: 'Closed forest, evergreen needle leaf',
        112: 'Closed forest, evergreen, broad leaf',
        113: 'Closed forest, deciduous needle leaf',
        114: 'Closed forest, deciduous broad leaf',
        115: 'Closed forest, mixed',
        116: 'Closed forest, unknown',
        121: 'Open forest, evergreen needle leaf',
        122: 'Open forest, evergreen broad leaf',
        123: 'Open forest, deciduous needle leaf',
        124: 'Open forest, deciduous broad leaf',
        125: 'Open forest, mixed',
        126: 'Open forest, unknown',
        20: 'Shrubs',
        30: 'Herbaceous vegetation',
        40: 'Cultivated and managed vegetation/agriculture',
        50: 'Urban / built up',
        60: 'Bare / sparse vegetation',
        70: 'Snow and Ice',
        80: 'Permanent water bodies',
        90: 'Herbaceous wetland',
        100: 'Moss and lichen',
        200: 'Open sea',
        # Add more LULC codes and their names as needed
    }
    # Find unique values in the data
    unique_values_in_data = np.unique(rds.values[0])
    # Plotting
    # Set the extent of the map
    if region and isinstance(region, (list, tuple)) and len(region) == 4:
        lon_min, lon_max, lat_min, lat_max = region
    else:
        lon_min, lon_max, lat_min, lat_max = np.min(longitude), np.max(longitude), np.min(latitude), np.max(latitude)

    # Plotting
    fig = plt.figure(figsize=(10, 8), dpi=150)
    
    # Define the map projection
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Set the extent to the Korean Peninsula
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    # Add country borders and coastlines
    ax.add_feature(cfeature.BORDERS, edgecolor='black')
    ax.coastlines()
    
    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Create the mesh plot using cropped_lons_2d and cropped_lats_2d
    mesh = plt.pcolormesh(longitude, latitude, rgb_image, shading='auto', transform=ccrs.PlateCarree())
    
    # Find unique values in the data
    unique_values_in_data = np.unique(rds.values[0])
    
    # Create legend patches only for the unique values present in the data
    legend_patches = [mpatches.Patch(color=np.array(color_map[code]) / 255, label=land_cover_classes[code]) for code in unique_values_in_data if code in color_map and code in land_cover_classes]
    
    # Plotting the RGB image with coordinates
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Add the legend to the plot
    # Position the legend below the figure and organize into two lines
    legend = ax.legend(handles=legend_patches, bbox_to_anchor=(0.5, -0.1), loc='upper center', borderaxespad=0, ncol=2, title='Land Cover Classes')
    
    plt.tight_layout()
    plt.show()


def plot_time_series(coords, longitude, latitude, data, label, x_label='Time', y_label='Soil Moisture'):
    """
    Adds a time series to an existing plot. If data is a 3D array, it finds the closest pixel for given coordinates and plots the time series for that pixel.
    If data is a 1D array, it plots the time series directly.
    
    Args:
    - coords: Tuple of (longitude, latitude).
    - longitude: 2D array of longitude values.
    - latitude: 2D array of latitude values.
    - data: 3D array or 1D array of data (e.g., SMAP data).
    - label: Label for the plot.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    """
    if data.ndim == 1:
        # Data is a 1D array; plot it directly
        time_series = data
    elif data.ndim == 3:
        # Data is a 3D array; find the closest pixel and extract its time series
        closest_pixel_index = Data.find_closest_index(longitude, latitude, coords)
        time_series = data[closest_pixel_index[0], closest_pixel_index[1], :]
    else:
        raise ValueError("Data must be either a 1D or 3D array.")

    plt.plot(time_series, '.-', label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

def create_gif_from_maps(nc_paths, domain_lon, domain_lat, variable_name, output_gif_path, start_index, end_index, padding, cmap='jet', duration=500, threshold_value=None, resampling=False, target_lon=False, target_lat=False):
    images = []
    global_min = np.inf
    global_max = -np.inf
    
    # Find the global min and max
    for i in range(start_index, end_index):
        data = get_variable_from_nc(nc_paths[i], variable_name)
        if threshold_value is not None:
            data[data < threshold_value] = np.nan  # Set values below the threshold to nan
        global_min = min(np.nanmin(data), global_min)
        global_max = max(np.nanmax(data), global_max)
    
    # Create each frame for the GIF
    for i in tqdm(range(start_index, end_index), desc="Processing"):
        data = get_variable_from_nc(nc_paths[i], variable_name)
        if threshold_value is not None:
            data[data < threshold_value] = np.nan  # Set values below the threshold to nan
        
        # Extract date from the filename using regex
        date_match = re.search(r'\d{8}', nc_paths[i])
        date_str = date_match.group(0) if date_match else 'Unknown Date'
        
        # Convert the date string to a more readable format if necessary
        # For example, '20150101' becomes '2015-01-01'
        formatted_date = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
        
        # Plot the data with fixed color scale and add the date as title
        if resampling:
            data = Resampling(target_lat, target_lon, domain_lat, domain_lon, data, 'nearest', 'mean', 1)
            fig, ax = plot_regional_map(target_lon, target_lat, data, variable_name, global_min, global_max, padding, cmap)
        else:
            fig, ax = plot_regional_map(domain_lon, domain_lat, data, variable_name, global_min, global_max, padding, cmap)

        ax.set_title(formatted_date)  # Set the extracted date as the title
        
        temp_img_path = f"temp_{i}.png"
        fig.savefig(temp_img_path)
        plt.close(fig)

        img = Image.open(temp_img_path)
        images.append(img)

        os.remove(temp_img_path)

    images[0].save(output_gif_path, save_all=True, append_images=images[1:], loop=0, duration=duration)


