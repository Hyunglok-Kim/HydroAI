import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import numpy as np
import os
import re
import rasterio

from PIL import Image

from hydroAI.LIS_LSM import get_variable_from_nc

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


def create_gif_from_maps(nc_paths, domain_lon, domain_lat, variable_name, output_gif_path, start_index, end_index, padding, cmap='jet', duration=500, threshold_value=None):
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
    for i in range(start_index, end_index):
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
        fig, ax = plot_regional_map(domain_lon, domain_lat, data, variable_name, global_min, global_max, padding, cmap)
        ax.set_title(formatted_date)  # Set the extracted date as the title
        
        temp_img_path = f"temp_{i}.png"
        fig.savefig(temp_img_path)
        plt.close(fig)

        img = Image.open(temp_img_path)
        images.append(img)

        os.remove(temp_img_path)

    images[0].save(output_gif_path, save_all=True, append_images=images[1:], loop=0, duration=duration)


