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

# Set the number of processors to use
num_processors = 10  # Adjust this based on your system's CPU resources

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

def initialize_with_empty_lists(shape):
    # Ensures that each element in the 2D list is an empty list
    return [[[] for _ in range(shape[1])] for _ in range(shape[0])]

def process_files(nc_files, ref_points, data_shape):
    local_angle_sum = np.zeros(data_shape, dtype=float)
    local_angle_sum_sq = np.zeros(data_shape, dtype=float)
    local_data_count = np.zeros(data_shape, dtype=int)
    local_timestamps = initialize_with_empty_lists(data_shape)

    for file_name in tqdm(nc_files, desc="Processing Files"):
        dataset = nc.Dataset(file_name)
        sp_lat = dataset.variables['sp_lat'][:].flatten().compressed()
        sp_lon = dataset.variables['sp_lon'][:].flatten().compressed() - 180
        sp_inc_angle = dataset.variables['sp_inc_angle'][:].flatten().compressed()
        time_units = dataset.variables['ddm_timestamp_utc'].units
        timestamp = (dataset.variables['ddm_timestamp_utc'][:].flatten().compressed() + cal_base_sec(time_units)).astype(int)
        timestamp = np.tile(timestamp, (4, ))

        sat_points = np.column_stack((sp_lat, sp_lon))
        tree = cKDTree(ref_points)
        _, indices = tree.query(sat_points)
        rows, cols = np.unravel_index(indices, data_shape)

        for row, col, angle, time in zip(rows, cols, sp_inc_angle, timestamp):
            local_angle_sum[row, col] += angle
            local_angle_sum_sq[row, col] += angle ** 2
            local_data_count[row, col] += 1
            local_timestamps[row][col].append(time)

    return local_angle_sum, local_angle_sum_sq, local_data_count, local_timestamps

def save_segments(data_array, num_segments, base_path, file_prefix):
    rows_per_segment = data_array.shape[0] // num_segments
    for seg_index in range(num_segments):
        start_row = seg_index * rows_per_segment
        end_row = start_row + rows_per_segment if seg_index < num_segments - 1 else data_array.shape[0]
        segment = data_array[start_row:end_row]
        filename = f"{base_path}/{file_prefix}_segment_{seg_index}.csv"
        np.savetxt(filename, segment, delimiter=',')

def calculate_and_save_timestamp_segments(timestamps, num_segments, base_path, file_prefix):
    data_shape = (len(timestamps), len(timestamps[0]))
    median_time_diffs = np.full(data_shape, np.nan, dtype=float)
    for i, row in enumerate(timestamps):
        for j, times in enumerate(row):
            if times:
                times = np.array(times)
                diffs = np.diff(np.sort(times))
                if diffs.size > 0:
                    median_time_diffs[i][j] = np.median(diffs)
    
    save_segments(median_time_diffs, num_segments, base_path, file_prefix)

def main():
    resol = '36km'
    base_dir = f"{cpuserver_data_FP}/CYGNSS/L1_V21"
    nc_file_list = list_nc_files(base_dir)
    ref_lon, ref_lat = hGrid.generate_lon_lat_e2grid(resol)
    data_shape = ref_lat.shape
    ref_points = np.column_stack((ref_lat.flatten(), ref_lon.flatten()))
    num_segments = 10  # Define number of spatial segments

    # Process all files to obtain aggregated data
    angle_sum, angle_sum_sq, data_count, timestamps = process_files(nc_file_list, ref_points, data_shape)

    # Save each segment of data arrays
    output_path = f"{cpuserver_data_FP}/CYGNSS/outputs"
    os.makedirs(output_path, exist_ok=True)
    save_segments(angle_sum, num_segments, output_path, 'angle_sum')
    save_segments(angle_sum_sq, num_segments, output_path, 'angle_sum_sq')
    save_segments(data_count, num_segments, output_path, 'data_count')
    calculate_and_save_timestamp_segments(timestamps, num_segments, output_path, 'timestamp_diffs')

    print("Processing and saving complete for all segments.")

if __name__ == '__main__':
    main()
