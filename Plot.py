import os
import os
import re

from scipy import ndimage
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
import matplotlib.ticker as ticker

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.gridliner as gridliner

import rasterio
from rasterio.plot import show
from tqdm import tqdm
from PIL import Image
import rioxarray
import imageio

from HydroAI.Data import get_variable_from_nc
from HydroAI.Data import Resampling
import HydroAI.Data as Data

# Set default font sizes using rcParams to ensure consistency
plt.rcParams['grid.linewidth'] = 1
plt.rcParams['axes.titlesize'] = 15  # Title size
plt.rcParams['axes.labelsize'] = 15  # Axis labels (x, y)
plt.rcParams['xtick.labelsize'] = 15 # X tick labels
plt.rcParams['ytick.labelsize'] = 15 # Y tick labels

def plot_map_old(longitude, latitude, values, title, cmin, cmax, cmap='jet', bounds=None, dem_path=None):
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
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Calculate the extent from the longitude and latitude
    extent = [longitude.min(), longitude.max(), latitude.min(), latitude.max()]
    
    # Set map extent if bounds are provided, else use the calculated extent
    if bounds:
        ax.set_extent(bounds, crs=ccrs.PlateCarree())
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Plot DEM as background if provided
    if dem_path:
        with rasterio.open(dem_path) as dem:
            dem_data = dem.read(1)  # Read the first band
            dem_extent = [dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top]
            ax.imshow(dem_data, origin='upper', extent=dem_extent, transform=ccrs.PlateCarree(), cmap='terrain', alpha=0.5)

    # Plot the data using imshow instead of pcolormesh
    #im = ax.imshow(values, transform=ccrs.PlateCarree(), cmap=cmap, vmin=cmin, vmax=cmax, extent=extent, origin='upper', interpolation='nearest')
    im = ax.pcolormesh(longitude, latitude, values, transform=ccrs.PlateCarree(), cmap=cmap, vmin=cmin, vmax=cmax)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
   
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = gridliner.LONGITUDE_FORMATTER
    gl.yformatter = gridliner.LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'black'}
    gl.ylabel_style = {'size': 15, 'color': 'black'}

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.5)
    cbar.set_label(title)
    im.set_clim(cmin, cmax)

    plt.show()
    
    return fig, ax

def plot_map(longitude, latitude, values, cmin, cmax, plot_title='title', label_title='values', cmap='jet', projection='Mollweide', bounds=None, dem_path=None, cbar_ticks=None, cbar_extend=None, points=None, save_fig_path=None):

    """
    Plots a map with the given data, either globally or within specified longitude and latitude bounds.

    Args:
    - longitude: 2D array of longitude values.
    - latitude: 2D array of latitude values.
    - values: 2D array of data values to plot.
    - title: Title for the colorbar and plot.
    - cmin, cmax: Minimum and maximum values for the colorbar.
    - cmap: Colormap to use for plotting data.
    - projection: Projection to use for the map. Defaults to 'Mollweide'.
        1.PlateCarree (most common)
        2.Mercator
        3.Miller
        4.Mollweide
        5.LambertCylindrical
        6.Robinson
        7.Sinusoidal
        8.InterruptedGoodeHomolosine
        9.Geostationary
        10.Orthographic
        11.NorthPolarStereo
        12.SouthPolarStereo
        13.AzimuthalEquidistant
        14.Gnomonic
        15.Stereographic
        16.LambertConformal
        17.AlbersEqualArea
        18.EquidistantConic
        19.LambertAzimuthalEqualArea
        (UTM is available but need to modify the code)
    - bounds: List or tuple of the format [lon_min, lon_max, lat_min, lat_max] for the map extent. If None, uses full range.
    - dem_path: Path to the DEM file for background in the plot (optional).
    - cbar_ticks: Colorbar ticks. Defaults to 'None' (Ex. [0, 0.5, 1])
    - cbar_extend: Colorbar extend. Defaults to 'None' (Ex. 'both')
    - points: Point indices to mark on the map. Defaults to 'None' (Ex. [(25, 10), (7, 35)])
    - save_fig_path: Path to save the figure. Defaults to 'None' (Ex. './test.png')

    Returns:
    - fig, ax: Figure and axes objects of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': getattr(ccrs, projection)()}, dpi=150)
    #fig.suptitle(plot_title, fontsize=16, y=0.67)
    #fig.suptitle(plot_title, fontsize=16, y=0.69)
    #fig.suptitle(plot_title, fontsize=16)
    #plt.tight_layout(rect=[0,0,1,0.95])
    #plt.subplots_adjust(top=0.64)
    #plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Calculate the extent from the longitude and latitude
    
    # Set map extent if bounds are provided, else use the calculated extent
    if bounds == 'global':
        ax.set_global()
    elif bounds == 'korea':
        bounds = [125.7, 129.7, 33.9, 38.8]
        ax.set_extent(bounds, crs=ccrs.PlateCarree())
    elif bounds:
        ax.set_extent(bounds, crs=ccrs.PlateCarree())
    else:
        extent = [longitude.min(), longitude.max(), latitude.min(), latitude.max()]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Plot DEM as background if provided
    if dem_path:
        with rasterio.open(dem_path) as dem:
            dem_data = dem.read(1)  # Read the first band
            dem_extent = [dem.bounds.left, dem.bounds.right, dem.bounds.bottom, dem.bounds.top]
            ax.imshow(dem_data, origin='upper', extent=dem_extent, transform=ccrs.PlateCarree(), cmap='terrain', alpha=0.5)

    # Plot the data using pcolormesh
    im = ax.pcolormesh(longitude, latitude, values, transform=ccrs.PlateCarree(), cmap=cmap, vmin=cmin, vmax=cmax)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = gridliner.LONGITUDE_FORMATTER
    gl.yformatter = gridliner.LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'black'}
    gl.ylabel_style = {'size': 15, 'color': 'black'}

    # Add colorbar and enforce color limits
    if cbar_extend == None:
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.5)
    else: # min, max, both
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.5, extend=cbar_extend)
    cbar.set_label(label_title)

    if cbar_ticks != None:
        cbar.set_ticks(cbar_ticks)

    # Ensure the color limits are set correctly on the image
    im.set_clim(cmin, cmax)

    # Mark the specified point if provided
    if points:
        for point in points:
            pixel_y, pixel_x = point
            lon = longitude[pixel_y, pixel_x]
            lat = latitude[pixel_y, pixel_x]
            ax.plot(lon, lat, marker='*', color='red', markersize=10, transform=ccrs.PlateCarree())

    if save_fig_path is not None:
        plt.tight_layout()
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight', transparent=True)

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
    ax.coastlines(resolution='10m')
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', edgecolor='black')
    gl = ax.gridlines(draw_labels=True, color='gray', linestyle='--', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 15, 'color': 'black'}
    gl.ylabel_style = {'size': 15, 'color': 'black'}

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

def plot_LULC_map_MCD12C1(longitude, latitude, values, lulc_type=1, title='MCD12C1 LULC map', projection='Mollweide', bounds=None, save_fig_path=None):
    """
    Plots MCD12C1 land cover data directly from given longitude, latitude, and LULC values,
    applying color mapping based on the LULC type.

    Args:
    longitude (numpy.ndarray): 2D array of longitude values.
    latitude (numpy.ndarray): 2D array of latitude values.
    values (numpy.ndarray): 2D array of LULC data values.
    lulc_type (int): The LULC type version (1, 2, or 3) for color mapping.
    title (str): Title for the plot.
    bounds (list): Geographic bounds as [lon_min, lon_max, lat_min, lat_max] for the map extent.
    """
    # Define color mappings for each LULC type
    class_names_and_colors = {
        1: {  # Type 1
            0: ('Water', '#4682B4'),
            1: ('Evergreen Needleleaf Forest', '#006400'),
            2: ('Evergreen Broadleaf Forest', '#228B22'),
            3: ('Deciduous Needleleaf Forest', '#8FBC8F'),
            4: ('Deciduous Broadleaf Forest', '#90EE90'),
            5: ('Mixed Forests', '#32CD32'),
            6: ('Closed Shrublands', '#FFD700'),
            7: ('Open Shrublands', '#FFA500'),
            8: ('Woody Savannas', '#FF8C00'),
            9: ('Savannas', '#BDB76B'),
            10: ('Grasslands', '#F0E68C'),
            11: ('Permanent Wetlands', '#E0FFFF'),
            12: ('Croplands', '#FFFFE0'),
            13: ('Urban and Built-up', '#D3D3D3'),
            14: ('Cropland/Natural Vegetation Mosaic', '#FAFAD2'),
            15: ('Snow and Ice', '#FFFFFF'),
            16: ('Barren or Sparsely Vegetated', '#A9A9A9')
        },
        2: {  # Type 2
            0: ('Water', '#4682B4'),
            1: ('Evergreen Needleleaf Forest', '#006400'),
            2: ('Evergreen Broadleaf Forest', '#228B22'),
            3: ('Deciduous Needleleaf Forest', '#8FBC8F'),
            4: ('Deciduous Broadleaf Forest', '#90EE90'),
            5: ('Mixed Forests', '#32CD32'),
            6: ('Closed Shrublands', '#FFD700'),
            7: ('Open Shrublands', '#FFA500'),
            8: ('Woody Savannas', '#FF8C00'),
            9: ('Savannas', '#BDB76B'),
            10: ('Grasslands', '#F0E68C'),
            12: ('Croplands', '#FFFFE0'),
            13: ('Urban and Built-up', '#D3D3D3'),
            15: ('Barren or Sparsely Vegetated', '#A9A9A9')
        },
        3: {  # Type 3
            0: ('Water', '#4682B4'),
            1: ('Grasses/Cereal', '#9ACD32'),
            2: ('Shrubs', '#8B4513'),
            3: ('Broadleaf Crops', '#32CD32'),
            4: ('Savannah', '#FAFAD2'),
            5: ('Evergreen Broadleaf Forest', '#006400'),
            6: ('Deciduous Broadleaf Forest', '#8FBC8F'),
            7: ('Evergreen Needleleaf Forest', '#228B22'),
            8: ('Deciduous Needleleaf Forest', '#90EE90'),
            9: ('Unvegetated', '#D3D3D3'),
            10: ('Urban', '#696969')
        }
    }

    # Create a colormap from the defined colors
    max_value = max(class_names_and_colors[lulc_type].keys())
    cmap_colors = [class_names_and_colors[lulc_type][i][1] for i in sorted(class_names_and_colors[lulc_type])]
    labels = [class_names_and_colors[lulc_type][i][0] for i in sorted(class_names_and_colors[lulc_type])]
    cmap = ListedColormap(cmap_colors)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': getattr(ccrs, projection)()}, dpi=150)
    # Create the plot
    #ax.suptitle(title, fontsize=16, y=0.67)
    fig.suptitle(title, fontsize=16, y=0.72)
    
    # Set geographic bounds if specified
    if bounds:
        ax.set_extent(bounds, crs=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='azure')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot the data
    mesh = ax.pcolormesh(longitude, latitude, values, cmap=cmap, transform=ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = gridliner.LONGITUDE_FORMATTER
    gl.yformatter = gridliner.LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'black'}
    gl.ylabel_style = {'size': 15, 'color': 'black'}
    
    # Add a colorbar with legend
    #cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, shrink=0.5)
    #cbar.set_label('Land Cover Type')
    #cbar.set_ticks(np.linspace(0, 1, len(labels)))
    #cbar.set_ticklabels(labels)

    # Create legend patches for detailed legend
    legend_patches = [mpatches.Patch(color=cmap_colors[i], label=labels[i]) for i in range(len(labels))]
    # Display the legend in 4 columns as requested
    legend = ax.legend(handles=legend_patches, title='Land Cover Classes', loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=3)

    if save_fig_path != None:
        plt.tight_layout()
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')

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
        date_match = re.search(r'\d{12}', nc_paths[i])
        date_str = date_match.group(0) if date_match else 'Unknown Date'
        
        # Convert the date string to a more readable format if necessary
        # For example, '20150101' becomes '2015-01-01'
        formatted_date = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}:{date_str[8:12]}'
        
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


def plot_kde_scatter(y_train_true, y_train_pred, y_test_true, y_test_pred):
    """
    This function creates a scatter plot to evaluate ML model performance with respect to train and test dataset.
    """
    def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
        """
        Scatter plot colored by 2d histogram
        (This code is made by 'Guilaume' in stackoverflow community.
        Ref. link: https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density/53865762#53865762)
        """
        if ax is None:
            fig, ax = plt.subplots()
        data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
        z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x,y]).T, method="splinef2d", bounds_error=False)

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        # Sort the points by density, so that the densest points are plotted last
        if sort:
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        sc = ax.scatter(x, y, c=z, **kwargs)
        norm = Normalize(vmin=np.min(z), vmax=np.max(z))
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis), ax=ax)

        return ax

    # Plotting training and test results with density scatter
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Training results
    density_scatter(y_train_true, y_train_pred, ax=axes[0], bins=[30, 30], cmap='viridis', label=f'MAE = {(np.mean(np.abs(y_train_true - y_train_pred))):.6f}')
    axes[0].plot([y_train_true.min(), y_train_true.max()], [y_train_true.min(), y_train_true.max()], 'k--', lw=4)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Train Data')
    axes[0].grid(True)
    axes[0].legend(loc='upper left', fontsize=20)

    # Test results
    density_scatter(y_test_true, y_test_pred, ax=axes[1], bins=[30, 30], cmap='viridis', label=f'MAE = {(np.mean(np.abs(y_test_true - y_test_pred))):.6f}')
    axes[1].plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], 'k--', lw=4)
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].set_title('Test Data')
    axes[1].grid(True)
    axes[1].legend(loc='upper left', fontsize=20)

    plt.tight_layout()
    plt.show()

def plot_kde_scatter_log_count(y_train_true, y_train_pred, y_test_true, y_test_pred, fontsize=20, global_min=None, global_max=None):
    plt.rc('font', family='Serif')

    if global_min == None:
        global_min = min(min(y_train_true), min(y_train_pred), min(y_test_true), min(y_test_pred))
    if global_max == None:
        global_max = max(max(y_train_true), max(y_train_pred), max(y_test_true), max(y_test_pred))

    """
    This function creates a scatter plot to evaluate ML model performance with respect to train and test dataset.
    """
    def bin_count_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
        """
        Scatter plot colored by the number of points in a bin with a threshold at 80% and exceeding values in yellow
        """
        if ax is None:
            fig, ax = plt.subplots()
        data, x_e, y_e = np.histogram2d(x, y, bins=bins)  # Calculate bin counts
        z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x,y]).T, method="splinef2d", bounds_error=False)

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        # Sort the points by number of points in each bin, so that the densest points are plotted last
        if sort:
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        # Calculate 80% threshold of bin counts
        max_z = np.max(z)
        threshold = 0.8 * max_z

        # Set up normalization with LogNorm and threshold
        norm = colors.LogNorm(vmin=1, vmax=threshold)  # vmin set to 1 to avoid issues with log(0)

        # Set up colormap
        cmap = cm.plasma
        cmap.set_over('yellow')

        # Scatter plot
        sc = ax.scatter(x, y, c=z, norm=norm, cmap=cmap, **kwargs)
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, extend='max')
        sc.set_clim(1, threshold)
        cbar.cmap.set_over('yellow')

        # Adjust the color bar ticks to represent bin counts in log scale
        log_ticks = np.logspace(np.log10(1), np.log10(threshold), num=5)
        cbar.set_ticks(log_ticks)
        cbar.set_ticklabels([f'{int(val)}' for val in log_ticks])
        cbar.ax.set_title('Bin Counts', fontsize=fontsize)

        return ax

    # Plotting training and test results with bin count scatter
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Define the ticks for x and y axis
    x_ticks = np.linspace(global_min, global_max, 5)
    y_ticks = np.linspace(global_min, global_max, 5)

    # Training results
    mae_train = np.mean(np.abs(y_train_true - y_train_pred))
    ax_train = bin_count_scatter(y_train_true, y_train_pred, ax=axes[0], bins=[30, 30])
    axes[0].plot([global_min, global_max], [global_min, global_max], 'k--', lw=2)
    axes[0].set_xlabel('Actual Values', fontsize=fontsize)
    axes[0].set_ylabel('Predicted Values', fontsize=fontsize)
    axes[0].set_title('Train Data', fontsize=fontsize)
    axes[0].grid(True)
    axes[0].set_xticks(x_ticks)
    axes[0].set_yticks(y_ticks)
    # Display MAE in upper left
    axes[0].text(0.05, 0.95, f'MAE = {mae_train:.6f}', transform=axes[0].transAxes,
                 fontsize=fontsize, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    # Test results
    mae_test = np.mean(np.abs(y_test_true - y_test_pred))
    ax_test = bin_count_scatter(y_test_true, y_test_pred, ax=axes[1], bins=[30, 30])
    axes[1].plot([global_min, global_max], [global_min, global_max], 'k--', lw=2)
    axes[1].set_xlabel('Actual Values', fontsize=fontsize)
    axes[1].set_ylabel('Predicted Values', fontsize=fontsize)
    axes[1].set_title('Test Data', fontsize=fontsize)
    axes[1].grid(True)
    axes[1].set_xticks(x_ticks)
    axes[1].set_yticks(y_ticks)
    # Display MAE in upper left
    axes[1].text(0.05, 0.95, f'MAE = {mae_test:.6f}', transform=axes[1].transAxes,
                 fontsize=fontsize, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

def generate_MP4_rotating_globe(nc_lon, nc_lat, data_3d, vmin, vmax, output_mp4, plot_title,
                                cbar_title=None, frame_duration=0.1, lons_center=np.arange(-180, 180, 10),
                                init_time_idx=0, time_step=1, time_length_per_rotation=1,
                                cmap='YlGnBu', background_color='white',
                                dpi=150, codec='libx264', bitrate='4000k'):
    """
    Generate a rotating Earth MP4 video of a 3D data array.

    Args:
    - nc_lon (numpy.ndarray): Longitude array of the data.
    - nc_lat (numpy.ndarray): Latitude array of the data.
    - data_3d (numpy.ndarray): 3D data array to be visualized.
    - vmin (float): Minimum value for the color scale.
    - vmax (float): Maximum value for the color scale.
    - output_mp4 (str): Path to save the output MP4 file.
    - plot_title (str): Title of the plot.
    - cbar_title (str): Title of the color bar.
    - frame_duration (float): Duration of each frame in seconds.
    - lons_center (numpy.ndarray): Longitude values at which to center the plot.
    - init_time_idx (int): Index of the initial time step.
    - time_step (int): Time step between frames.
    - time_length_per_rotation (int): Number of time steps per rotation.
    - cmap (str): Colormap to use for the plot.
    - background_color (str): Background color of the plot.
    - dpi (int): DPI of the plot.
    - codec (str): Codec to use for the MP4 file.
    - bitrate (str): Bitrate of the MP4 file.
    """
    fps = max(1, int(round(1.0 / frame_duration)))
    writer = imageio.get_writer(
        output_mp4, fps=fps, codec=codec, bitrate=bitrate, pixelformat='yuv420p'
    )
    if background_color == 'black': text_color = 'white'
    else: text_color = 'black'

    time_idx = init_time_idx
    try:
        for lon0 in tqdm(lons_center, desc="Rendering frames (MP4)"):
            for _ in range(time_length_per_rotation):
                data = data_3d[:, :, time_idx]

                fig = plt.figure(figsize=(6, 6), dpi=dpi, facecolor=background_color)
                ax = plt.axes(
                    projection=ccrs.Orthographic(central_longitude=lon0, central_latitude=20),
                    facecolor=background_color
                )
                ax.set_global()
                ax.coastlines(color='black', linewidth=0.5)
                ax.stock_img()

                mesh = ax.pcolormesh(
                    nc_lon, nc_lat, data,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap, vmin=vmin, vmax=vmax,
                )

                if cbar_title is not None:
                    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                                        fraction=0.046, pad=0.04, extend='both')
                    cbar.set_label(cbar_title, color=text_color)
                    cbar.ax.tick_params(labelcolor=text_color)

                ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', linestyle='--')
                ax.set_title(plot_title, fontsize=20, color=text_color)

                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                try:
                    buf = fig.canvas.tostring_rgb()
                except AttributeError: # Due to the version of matplotlib
                    buf = fig.canvas.buffer_rgba()
                    buf = np.asarray(buf)  # ensures numpy array if needed (for older versions)
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, -1)[..., :3]
                writer.append_data(frame)
                plt.close(fig)

                time_idx += time_step

    finally:
        writer.close()

    print(f"MP4 saved to {output_mp4} (fps={fps}, codec={codec})")
