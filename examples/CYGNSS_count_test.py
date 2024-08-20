import os
import sys
import platform
import importlib
from multiprocessing import Pool
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
from tqdm import tqdm
import re
from datetime import datetime

# Configuration based on the operating system
if platform.system() == 'Darwin':  # macOS
    base_FP = '/Users/hyunglokkim/Insync/hkim@geol.sc.edu/Google_Drive'
    cpuserver_data_FP = '/Users/hyunglokkim/cpuserver_data'
else:
    base_FP = '/data'
    cpuserver_data_FP = '/data'

sys.path.append(base_FP + '/python_modules')
import HydroAI.Grid as hGrid
importlib.reload(hGrid)

num_processors = 10  # Set the number of processors for multiprocessing
num_segments = 10    # Number of spatial segments

def list_nc_files(base_dir):
    nc_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".nc4"):
                nc_files.append(os.path.join(root, file))
    nc_files.sort()
    return nc_files

def cal_base_sec(time_units):
    match = re.search(r'seconds since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6})', time_units)
    if match:
        base_date_str = match.group(1)
        base_datetime = datetime.strptime(base_date_str, '%Y-%m-%d %H:%M:%S.%f')
        epoch_datetime = datetime(2017, 1, 1)
        return int((base_datetime - epoch_datetime).total_seconds())
    return 0

def process_segment(args):
    segment_index, file_list, ref_points, segment_bounds = args
    local_data = np.zeros((segment_bounds[1] - segment_bounds[0], len(ref_points[0])), dtype=float)  # Example of data structure

    # Simulate some processing
    for _ in range(5):  # Dummy loop to represent file processing
        local_data += np.random.rand(segment_bounds[1] - segment_bounds[0], len(ref_points[0]))

    # Simulating delay
    from time import sleep
    sleep(1)  # Simulate time-consuming computation

    return segment_index, local_data

def save_segment(segment_data, segment_index, output_path):
    filename = os.path.join(output_path, f"segment_{segment_index}.csv")
    np.savetxt(filename, segment_data, delimiter=',')
    print(f"Segment {segment_index} processed and saved.")

def main():
    resol = '36km'
    base_dir = f"{cpuserver_data_FP}/CYGNSS/L1_V21"
    nc_file_list = list_nc_files(base_dir)
    ref_lon, ref_lat = hGrid.generate_lon_lat_e2grid(resol)
    data_shape = (len(ref_lat), len(ref_lon))
    ref_points = np.column_stack((ref_lat.flatten(), ref_lon.flatten()))

    # Define segments
    rows_per_segment = data_shape[0] // num_segments
    segment_bounds = [(i * rows_per_segment, min((i + 1) * rows_per_segment, data_shape[0])) for i in range(num_segments)]

    # Prepare arguments for multiprocessing
    segment_args = [(i, nc_file_list, ref_points, bounds) for i, bounds in enumerate(segment_bounds)]

    output_path = f"{cpuserver_data_FP}/CYGNSS/outputs"
    os.makedirs(output_path, exist_ok=True)

    # Use multiprocessing to process each segment
    with Pool(num_processors) as pool:
        for segment_index, segment_data in tqdm(pool.imap_unordered(process_segment, segment_args), total=len(segment_args), desc="Processing Segments"):
            save_segment(segment_data, segment_index, output_path)

if __name__ == '__main__':
    main()

