import rasterio
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.gridliner as gridliner

class SentinelBandReader:
    def __init__(self, folder_path, product='S30'):
        self.folder_path = folder_path
        self.scale_factor = 0.0001
        self.fill_value = -9999
        # Define the mapping of band nicknames to file names
        if product == 'S30':
            self.band_files = {
                'coastal': 'B01_merged_WGS84.tif',  # Coastal/aerosol band
                'blue': 'B02_merged_WGS84.tif',     # Blue band
                'green': 'B03_merged_WGS84.tif',    # Green band
                'red': 'B04_merged_WGS84.tif',      # Red band
                'red_edge1': 'B05_merged_WGS84.tif',# Red edge 1
                'red_edge2': 'B06_merged_WGS84.tif',# Red edge 2
                'red_edge3': 'B07_merged_WGS84.tif',# Red edge 3
                'nir': 'B08_merged_WGS84.tif',      # Near-infrared
                'red_edge4': 'B8A_merged_WGS84.tif',# Red edge 4
                'water_vapor': 'B09_merged_WGS84.tif', # Water vapor
                'cirrus': 'B10_merged_WGS84.tif',   # Cirrus
                'swir1': 'B11_merged_WGS84.tif',    # Short-wave infrared 1
                'swir2': 'B12_merged_WGS84.tif',    # Short-wave infrared 2
                'qa': 'Fmask_merged_WGS84.tif'       # QA band
            }
        else: # product = 'L30'
            self.band_files = {
                'coastal': 'B01_merged_WGS84.tif',  # Coastal/aerosol band
                'blue': 'B02_merged_WGS84.tif',     # Blue band
                'green': 'B03_merged_WGS84.tif',    # Green band
                'red': 'B04_merged_WGS84.tif',      # Red band
                'nir': 'B05_merged_WGS84.tif',# Near infrared
                'swir1': 'B06_merged_WGS84.tif',#  Short-wave infrared 1
                'swir2': 'B07_merged_WGS84.tif',# Short-wave infrared 2
                'cirrus': 'B09_merged_WGS84.tif', # Cirrus
                'tir1': 'B10_merged_WGS84.tif',   # Thermal Infrared 1
                'tir2': 'B11_merged_WGS84.tif',    # Thermal Infrared 2
                'qa': 'Fmask_merged_WGS84.tif'       # QA band
            }
            
    def get_band_with_transform(self, band):
        """Return band data along with its affine transform."""
        band_file = self.band_files.get(band)
        if not band_file:
            raise ValueError(f"No such band: {band}")
        path = f"{self.folder_path}/{band_file}"
        with rasterio.open(path) as src:
            data = src.read(1).astype('float32')
            # Apply fill value and scaling factor for non-atmospheric correction bands
            if band not in ['water_vapor', 'cirrus', 'coastal', 'qa']:
                data = np.where(data == self.fill_value, np.nan, data * self.scale_factor)
                data = np.where((data < 0) | (data > 1), np.nan, data)
            return data, src.transform
            
    def _read_band(self, band_file):
        """Read and process a single band file."""
        path = f"{self.folder_path}/{band_file}"
        with rasterio.open(path) as src:
            band_data = src.read(1).astype('float32')
            # Apply fill value and scaling factor for non-atmospheric correction bands
            if 'water_vapor' not in band_file and 'cirrus' not in band_file and 'coastal' not in band_file and 'qa' not in band_file:
                processed_data = np.where(band_data == self.fill_value, np.nan, band_data * self.scale_factor)
                processed_data = np.where((processed_data < 0) | (processed_data > 1), np.nan, processed_data)
            else:
                processed_data = band_data
        return processed_data

    def get_lat_lon_arrays(self, band):
        """Generate latitude and longitude arrays for the specified band."""
        band_file = self.band_files.get(band)
        if not band_file:
            raise ValueError(f"No such band: {band}")
        path = f"{self.folder_path}/{band_file}"
        with rasterio.open(path) as src:
            width, height = src.width, src.height
            transform = src.transform

            # Generate pixel coordinates
            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            xs, ys = rasterio.transform.xy(transform, rows, cols)

            # xs and ys are 2D arrays containing all longitude and latitude values respectively
            return np.array(xs), np.array(ys)
            
    def __getattr__(self, band):
        """Generic getter to fetch band data based on band name."""
        band_file = self.band_files.get(band)
        if band_file:
            return self._read_band(band_file)
        else:
            raise AttributeError(f"No such band: {band}")
#------------------------------------------------------#
# Example usage:
#s2_reader = SentinelBandReader(base_folder)

# Access different bands
#red_data = s2_reader.red  # Red band
#nir_data = s2_reader.nir  # Near-infrared band
#swir1_data = s2_reader.swir1  # Short-wave infrared 1
#qa_data = s2_reader.qa  # QA band

#print("Red Band Data:")
#print(red_data)
#------------------------------------------------------#

class SentinelBandPlotter:
    def __init__(self, band_reader):
        self.band_reader = band_reader

    def plot_band(self, band, cmap='gray', title=None):
        data, transform = self.band_reader.get_band_with_transform(band)
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        
        height, width = data.shape
        west, north = transform * (0, 0)
        east, south = transform * (width, height)
        extent = [west, east, south, north]

        img = ax.imshow(data, cmap=cmap, extent=extent, origin='upper', interpolation='none')
        plt.colorbar(img, ax=ax, orientation='vertical', label='Normalized Reflectance')
        ax.add_feature(cfeature.COASTLINE)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = gridliner.LONGITUDE_FORMATTER
        gl.yformatter = gridliner.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}
        
        if title:
            plt.title(title)
        plt.show()

    def plot_rgb(self, title=None):
        # Standard band identifiers for Sentinel-2 RGB composite
        red_band = 'red'
        green_band = 'green'
        blue_band = 'blue'

        # Retrieve data and transforms for each band
        red, transform = self.band_reader.get_band_with_transform(red_band)
        green, _ = self.band_reader.get_band_with_transform(green_band)
        blue, _ = self.band_reader.get_band_with_transform(blue_band)
        
        # Normalize each band data to [0, 1] range
        red_normalized = self._normalize_band(red)
        green_normalized = self._normalize_band(green)
        blue_normalized = self._normalize_band(blue)
        
        # Stack bands into an RGB image
        rgb = np.stack([red_normalized, green_normalized, blue_normalized], axis=-1)
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        height, width = red.shape
        west, north = transform * (0, 0)
        east, south = transform * (width, height)
        extent = [west, east, south, north]
        
        ax.imshow(rgb, extent=extent, origin='upper')
        ax.add_feature(cfeature.COASTLINE)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = gridliner.LONGITUDE_FORMATTER
        gl.yformatter = gridliner.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}
        
        if title:
            plt.title(title)
        plt.show()

    def plot_false_color(self, title=None):

        # Standard band identifiers for Sentinel-2 RGB composite
        red_band = 'red'
        green_band = 'green'
        nir_band = 'nir'

        # Retrieve data and transforms for each band
        red, transform = self.band_reader.get_band_with_transform(red_band)
        green, _ = self.band_reader.get_band_with_transform(green_band)
        nir, _ = self.band_reader.get_band_with_transform(nir_band)

        # Replace NaNs with zero and infinite values with finite maximums of each array
        nir = np.nan_to_num(nir, nan=0, posinf=np.nanmax(nir[np.isfinite(nir)]), neginf=0)
        red = np.nan_to_num(red, nan=0, posinf=np.nanmax(red[np.isfinite(red)]), neginf=0)
        green = np.nan_to_num(green, nan=0, posinf=np.nanmax(green[np.isfinite(green)]), neginf=0)

        # Normalize each band data to [0, 1] range
        false_color_image = np.stack((nir, red, green), axis=-1)
        # Scale the data to [0, 1] if not already scaled
        # Normalize the data to [0, 1] if not already normalized
        max_value = np.max(false_color_image)
        if max_value > 1:
            false_color_image = false_color_image / max_value
    
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        height, width = red.shape
        west, north = transform * (0, 0)
        east, south = transform * (width, height)
        extent = [west, east, south, north]
        
        ax.imshow(false_color_image, extent=extent, origin='upper')
        ax.add_feature(cfeature.COASTLINE)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = gridliner.LONGITUDE_FORMATTER
        gl.yformatter = gridliner.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}
        
        if title:
            plt.title(title)
        plt.show()
    
    def plot_index(self, index_variable, transform, threshold=None, cmap='viridis', title=None, bounds=None):
        
        data = index_variable.astype(float)  # Convert to floating-point type
    
        if threshold is not None:
            # Handle NaN values
            nan_mask = np.isnan(data)
            data[nan_mask] = 0  # Set NaN values to zero initially
            # Apply thresholding
            mask_1 = (data < threshold) & (~nan_mask)
            mask_2 = (data >= threshold) & (~nan_mask)
            data[mask_1] = 0.5
            data[mask_2] = 1
    
            # Define custom colormap for thresholded data
            colors = ['black', 'grey', 'blue']
            labels = ["Invalid data", "Valid data but possibly not water", "Possibly water"]
            cmap = mcolors.ListedColormap(colors)
            norm = mcolors.BoundaryNorm([0, 0.5, 1, 2], cmap.N)
        else:
            # Use the specified cmap and define a norm that covers the usual data range
            norm = plt.Normalize(np.nanmin(data), np.nanmax(data))
            cmap = plt.get_cmap(cmap)
            cmap.set_bad(color='black')  # Set NaN values to be black

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
        # Set map extent if bounds are provided
        if bounds:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
    
        height, width = data.shape
        west, north = transform * (0, 0)
        east, south = transform * (width, height)
        extent = [west, east, south, north]
    
        img = ax.imshow(data, cmap=cmap, extent=extent, origin='upper', interpolation='none', norm=norm)
        ax.add_feature(cfeature.COASTLINE)
    
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = gridliner.LONGITUDE_FORMATTER
        gl.yformatter = gridliner.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'black'}
        gl.ylabel_style = {'size': 15, 'color': 'black'}
    
        # Handle legend creation based on threshold presence
        if threshold is not None:
            patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
            legend = ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                               fancybox=True, shadow=False, ncol=3)
        else:
            cbar = plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.1)
            cbar.set_label('Data Value Scale')
    
        if title:
            plt.title(title)
    
        plt.show()
        
    def _normalize_band(self, band_data):
        """Normalize band data to [0, 1] range."""
        nan_mask = np.isnan(band_data)
        band_min, band_max = np.nanmin(band_data), np.nanmax(band_data)
        norm_band = (band_data - band_min) / (band_max - band_min)
        norm_band[nan_mask] = 0
        return norm_band
#------------------------------------------------------#
# Example usage 
#s2_reader = SentinelBandReader(base_folder)
#s2_plotter = SentinelBandPlotter(s2_reader)
#s2_plotter.plot_band('red', 'jet', 'Red band')
#s2_plotter.plot_rgb('RGB image')
#------------------------------------------------------#

class WaterIndicesCalculator:
    def __init__(self, band_reader):
        self.band_reader = band_reader

    def calculate_mndwi(self):
        """Calculate Modified Normalized Difference Water Index (MNDWI)."""
        green, transform = self.band_reader.get_band_with_transform('green')
        swir1, _ = self.band_reader.get_band_with_transform('swir1')
        
        # Set MNDWI values to NaN where denominator is zero or close to zero
        denominator = green + swir1
        with np.errstate(divide='ignore', invalid='ignore'):
            mndwi = (green - swir1) / denominator
            mndwi[denominator == 0] = np.nan  # Explicitly set where denominator is zero to NaN
    
        return mndwi, transform
#------------------------------------------------------#
# Example usage
#s2_reader = SentinelBandReader(base_folder)
#water_indices_calculator = WaterIndicesCalculator(s2_reader)
#mndwi, transform = water_indices_calculator.calculate_mndwi()

#s2_plotter = SentinelBandPlotter(s2_reader)
#bounds = [38.895, 39.445, 21.445, 22.995]
#s2_plotter.plot_index(mndwi, transform, threshold=0, title='Modified Normalized Difference Water Index (MNDWI)', bounds = bounds)
#s2_plotter.plot_index(mndwi, transform, title='Modified Normalized Difference Water Index (MNDWI)', cmap='jet_r', bounds = bounds)
#------------------------------------------------------#
