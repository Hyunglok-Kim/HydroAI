"""
Count CYGNSS L1 specular-point observations on an EASE2 grid.

Merged from the former CYGNSS_count.py / CYGNSS_count_seg.py pair:
segmented processing is the default; pass --segment-size 0 to process
everything in one batch, and --plot for a quick visual check.

Example:
    python CYGNSS_count.py --resol 36km --segment-size 0 --max-files 301 --plot
    python CYGNSS_count.py --resol 3km
"""
import argparse
import os
import re
import sys
import platform
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
from tqdm import tqdm

# Check the platform to set file paths
if platform.system() == 'Darwin':  # macOS
    base_FP = '/Users/hyunglokkim/Insync/hkim@geol.sc.edu/Google_Drive'
    cpuserver_data_FP = '/Users/hyunglokkim/cpuserver_data'
else:
    base_FP = '/data'
    cpuserver_data_FP = '/data'

# Add Python modules path and import
sys.path.append(base_FP + '/python_modules')
import HydroAI.Grid as hGrid


def list_nc_files(base_dir):
    nc_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".nc4"):
                full_path = os.path.join(root, file)
                nc_files.append(full_path)
    nc_files.sort()
    return nc_files


def cal_base_sec(time_units):
    match = re.search(r'seconds since (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6})', time_units)
    if match:
        original_base_date_str = match.group(1)

    original_base_datetime = datetime.strptime(original_base_date_str, '%Y-%m-%d %H:%M:%S.%f')
    new_base_datetime = datetime(2017, 1, 1)
    delta_seconds = (original_base_datetime - new_base_datetime).total_seconds()
    return delta_seconds


def initialize_with_empty_lists(shape):
    data = np.empty(shape, dtype=object)
    for (r, c), _ in np.ndenumerate(data):
        data[r, c] = []
    return data


def clear_lists(arr):
    clear_func = np.frompyfunc(lambda x: x.clear(), 1, 1)
    clear_func(arr)


def process_files(file_names, ref_points, data_shape):
    local_angle_sum = np.zeros(data_shape, dtype=float)
    local_angle_sum_sq = np.zeros(data_shape, dtype=float)
    local_data_count = np.zeros(data_shape, dtype=int)

    local_timestamp_all_update = initialize_with_empty_lists(data_shape)
    local_timestamp_median = initialize_with_empty_lists(data_shape)
    tree = cKDTree(ref_points)

    for file_name in tqdm(file_names, desc="Processing Files", leave=False):
        dataset = nc.Dataset(file_name)
        sp_lat = dataset.variables['sp_lat'][:].flatten().compressed()
        sp_lon = dataset.variables['sp_lon'][:].flatten().compressed() - 180
        sp_inc_angle = dataset.variables['sp_inc_angle'][:].flatten().compressed()

        time_units = dataset.variables['ddm_timestamp_utc'].units  # since xx
        timestamp = dataset.variables['ddm_timestamp_utc'][:].flatten().compressed() + cal_base_sec(time_units)
        timestamp = np.tile(timestamp, (4, ))

        sat_points = np.column_stack((sp_lat, sp_lon))
        _, indices = tree.query(sat_points)
        rows, cols = np.unravel_index(indices, data_shape)

        clear_lists(local_timestamp_all_update)
        for row, col, angle, time in zip(rows, cols, sp_inc_angle, timestamp):
            local_angle_sum[row, col] += angle
            local_angle_sum_sq[row, col] += angle ** 2
            local_data_count[row, col] += 1
            local_timestamp_all_update[row, col].append(time)

        for r in range(local_timestamp_all_update.shape[0]):
            for c in range(local_timestamp_all_update.shape[1]):
                if len(local_timestamp_all_update[r, c]) >= 1:
                    local_timestamp_median[r, c].append(np.median(local_timestamp_all_update[r, c]))

    return local_angle_sum, local_angle_sum_sq, local_data_count, local_timestamp_median


def aggregate_and_save(results, ref_lat, data_shape, output_dir, resol, tag):
    final_angle_sum = np.sum([result[0] for result in results], axis=0)
    final_angle_sum_sq = np.sum([result[1] for result in results], axis=0)
    final_data_count = np.sum([result[2] for result in results], axis=0)
    final_timestamp_median = initialize_with_empty_lists(ref_lat.shape)

    for result in results:
        local_timestamp_median = result[3]
        for r in range(data_shape[0]):
            for c in range(data_shape[1]):
                final_timestamp_median[r, c].extend(local_timestamp_median[r, c])

    median_time_differences = np.full(data_shape, np.nan, dtype=float)
    for r in range(final_timestamp_median.shape[0]):
        for c in range(final_timestamp_median.shape[1]):
            if len(final_timestamp_median[r, c]) > 1:
                final_timestamp_median[r, c].sort()
                differences = np.diff(final_timestamp_median[r, c])
                median_time_differences[r, c] = np.median(differences)

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, f"CYGNSS_angle_sum_{resol}{tag}.csv"), final_angle_sum, delimiter=',')
    np.savetxt(os.path.join(output_dir, f"CYGNSS_angle_sum_sq_{resol}{tag}.csv"), final_angle_sum_sq, delimiter=',')
    np.savetxt(os.path.join(output_dir, f"CYGNSS_data_count_{resol}{tag}.csv"), final_data_count, delimiter=',')
    np.savetxt(os.path.join(output_dir, f"CYGNSS_median_time_diff_{resol}{tag}.csv"), median_time_differences, delimiter=',')

    return final_data_count, median_time_differences


def plot_median_time_diff(median_time_differences):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    t = median_time_differences / 60 / 60
    im = plt.imshow(t, vmin=np.nanpercentile(t, 20), vmax=np.nanpercentile(t, 80), cmap='viridis')
    plt.colorbar(im)
    plt.title("Median revisit time (hours)")
    plt.xlabel("Longitude Index")
    plt.ylabel("Latitude Index")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Count CYGNSS L1 observations on an EASE2 grid.")
    parser.add_argument('--resol', default='36km', help="EASE2 grid resolution key (e.g. 36km, 25km, 9km, 3km)")
    parser.add_argument('--segment-size', type=int, default=300,
                        help="Files per segment; 0 processes all files in one batch")
    parser.add_argument('--max-files', type=int, default=0, help="Limit number of input files (0 = all)")
    parser.add_argument('--num-processes', type=int, default=os.cpu_count(), help="Worker process count")
    parser.add_argument('--input-dir', default=cpuserver_data_FP + "/CYGNSS/L1_V21")
    parser.add_argument('--output-dir', default=cpuserver_data_FP + "/CYGNSS/data_counts_csv")
    parser.add_argument('--plot', action='store_true', help="Plot median revisit time when done")
    args = parser.parse_args()

    nc_file_list = list_nc_files(args.input_dir)
    if args.max_files:
        nc_file_list = nc_file_list[:args.max_files]

    ref_lon, ref_lat = hGrid.generate_lon_lat_e2grid(args.resol)
    data_shape = ref_lat.shape
    ref_points = np.column_stack((ref_lat.flatten(), ref_lon.flatten()))

    segment_size = args.segment_size or len(nc_file_list)
    for segment_index, start in enumerate(range(0, len(nc_file_list), segment_size)):
        current_file_list = nc_file_list[start:start + segment_size]
        num_processes = min(args.num_processes, len(current_file_list))
        chunk_size = len(current_file_list) // num_processes + (len(current_file_list) % num_processes > 0)

        print(f'Calculating.... Segment {segment_index + 1}')
        pool = Pool(processes=num_processes)
        results = pool.starmap(process_files, [(current_file_list[i:i + chunk_size], ref_points, data_shape)
                                               for i in range(0, len(current_file_list), chunk_size)])
        pool.close()
        pool.join()

        print('Aggregating Results....')
        tag = f"_seg_{segment_index + 1}" if args.segment_size else ""
        final_data_count, median_time_differences = aggregate_and_save(
            results, ref_lat, data_shape, args.output_dir, args.resol, tag)
        print(f'Finished segment {segment_index + 1}. Data count shape:', final_data_count.shape)

        if args.plot:
            plot_median_time_diff(median_time_differences)


if __name__ == '__main__':
    main()
