import os
import os
import re

from scipy import ndimage
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.gridliner as gridliner

import rasterio
from rasterio.plot import show
from tqdm import tqdm
from PIL import Image
import rioxarray

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

def plot_map(longitude, latitude, values, cmin, cmax, plot_title='title', label_title='values', cmap='jet', projection='Mollweide', bounds=None, dem_path=None):
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

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.5)
    cbar.set_label(label_title)
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

def plot_LULC_map_MCD12C1(longitude, latitude, values, lulc_type=1, title='MCD12C1 LULC map', projection='Mollweide', bounds=None):
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


