import numpy as np
from scipy import ndimage
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import rioxarray
import os

# Define the majority filter function
def majority_filter(data, size=3, nodata=-9999):
    """
    Apply a majority filter to the input data.
    Args:
    - data: 2D array of data to be filtered.
    - size: The size of the neighborhood used for the filter.
    - nodata: Value to represent no-data in the input.
    
    Returns:
    - Filtered data as a 2D array.
    """
    def filter_func(x):
        x = x[x != nodata]
        x = x[x >= 0]
        x = x.astype(int)
        if len(x) == 0:
            return nodata
        else:
            counts = np.bincount(x)
            return np.argmax(counts)
    return ndimage.generic_filter(data, filter_func, size=size)

def copernicus(FP, input_file, dst_crs, resolution, output_FP=None):
    """
    Reproject and resample the input raster data, then apply the majority filter to the data.
    Args:
    - FP: File path to the input raster.
    - input_file: Name of the input raster file.
    - dst_crs: The target coordinate reference system.
    - resolution: The target resolution for resampling.
    - output_FP: File path where the output file will be saved. If None, the output file is created temporarily and removed.
    
    Returns:
    - rds: Reprojected and resampled data.
    - lon: Longitude values of pixel centers.
    - lat: Latitude values of pixel centers.
    """
    input_file_path = f"{FP}/{input_file}"
    base_name = input_file.split('.')[0]
    crs_name = dst_crs.split(':')[1]

    # Construct the output file path
    output_file = f"{output_FP}/{base_name}_{crs_name}_res{resolution}.tif" if output_FP else f"{base_name}_{crs_name}_res{resolution}_temp.tif"
    
    with rasterio.open(input_file_path) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height, 'dtype': 'int32'})
        
        with rasterio.open(output_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i), src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.nearest)
    
    rds = rioxarray.open_rasterio(output_file)
    rds = rds.rio.reproject(dst_crs, resolution=resolution)
    
    for i in range(rds.rio.count):
        rds.values[i] = majority_filter(rds.values[i], size=3, nodata=-9999)
    
    if not output_FP:
        # If an output file path was not specifically provided, remove the temporary file
        os.remove(output_file)
    
    x, y = np.meshgrid(np.arange(rds.rio.width), np.arange(rds.rio.height))
    x, y = rds.rio.transform() * (x, y)
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    lon_flat, lat_flat = rasterio.warp.transform(src_crs=rds.rio.crs, dst_crs='EPSG:4326', xs=x_flat, ys=y_flat)
    lon_flat = np.array(lon_flat)
    lat_flat = np.array(lat_flat)
    
    lon = lon_flat.reshape(x.shape)
    lat = lat_flat.reshape(y.shape)

    return rds, lon, lat

