import os
import sys
import platform
import importlib
from multiprocessing import Pool
import numpy as np
import netCDF4 as nc
from scipy.spatial import cKDTree
from tqdm import tqdm
import matplotlib.pyplot as plt

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
importlib.reload(hGrid)

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

    #timestamp_all_ddm = initialize_with_empty_lists(ref_lat.shape)
    local_timestamp_all_update = initialize_with_empty_lists(ref_lat.shape)
    local_timestamp_median = initialize_with_empty_lists(ref_lat.shape)
    tree = cKDTree(ref_points)
    
    for file_name in tqdm(file_names, desc="Processing Files", leave=False):
        dataset = nc.Dataset(file_name)
        sp_lat = dataset.variables['sp_lat'][:].flatten().compressed()
        sp_lon = dataset.variables['sp_lon'][:].flatten().compressed() - 180
        sp_inc_angle = dataset.variables['sp_inc_angle'][:].flatten().compressed()
        
        time_units = dataset.variables['ddm_timestamp_utc'].units #since xx
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
                if len(local_timestamp_all_update[r, c]) >= 1:  # Ensure there are enough timestamps to calculate a difference
                    local_timestamp_median[r,c].append(np.median(local_timestamp_all_update[r, c]))
                    
    return local_angle_sum, local_angle_sum_sq, local_data_count, local_timestamp_median

def main():
    base_dir = cpuserver_data_FP+"/CYGNSS/L1_V21"
    nc_file_list = list_nc_files(base_dir)
    resol = '3km'
    ref_lon, ref_lat = hGrid.generate_lon_lat_e2grid(resol)
    
    data_shape = ref_lat.shape
    ref_points = np.column_stack((ref_lat.flatten(), ref_lon.flatten()))

    num_processes = 180
    chunk_size = len(nc_file_list) // num_processes + (len(nc_file_list) % num_processes > 0)
    
    pool = Pool(processes=num_processes)
    results = pool.starmap(process_files, [(nc_file_list[i:i + chunk_size], ref_points, data_shape) for i in range(0, len(nc_file_list), chunk_size)])
    pool.close()
    pool.join()
    
    # Aggregate results
    final_angle_sum = np.sum([result[0] for result in results], axis=0)
    final_angle_sum_sq = np.sum([result[1] for result in results], axis=0)
    final_data_count = np.sum([result[2] for result in results], axis=0)
    final_timestamp_median = initialize_with_empty_lists(ref_lat.shape)

    # Concatenate timestamp lists from each process
    for result in results:
        local_timestamp_median = result[3]
        for r in range(data_shape[0]):
            for c in range(data_shape[1]):
                final_timestamp_median[r, c].extend(local_timestamp_median[r, c])

    median_time_differences = np.empty(data_shape, dtype=float)
    for r in range(final_timestamp_median.shape[0]):
        for c in range(final_timestamp_median.shape[1]):
            if len(fianl_timestamp_median[r, c]) > 1:  # Ensure there are enough timestamps to calculate a difference
                final_timestamp_median[r, c].sort()  # Sort timestamps in place
                
                # Calculate differences and the average
                differences = np.diff(final_timestamp_median[r, c])
                median_diff = np.median(differences)
                median_time_differences[r, c] = median_diff
            else:
                median_time_differences[r, c] = None  # No data to process
    
    # Save to CSV
    np.savetxt(f"/data/CYGNSS/data_counts_csv/CYGNSS_angle_sum_{resol}.csv", final_angle_sum, delimiter=',')
    np.savetxt(f"/data/CYGNSS/data_counts_csv/CYGNSS_angle_sum_sq_{resol}.csv", final_angle_sum_sq, delimiter=',')
    np.savetxt(f"/data/CYGNSS/data_counts_csv/CYGNSS_data_count_{resol}.csv", final_data_count, delimiter=',')    
    np.savetxt(f"/data/CYGNSS/data_counts_csv/CYGNSS_median_time_diff_{resol}.csv", median_time_differences, delimiter=',')    

    # Optional: Plotting the results for visual confirmation
    plt.figure(figsize=(10, 6))
    im = plt.imshow(median_time_differences, cmap='viridis')
    plt.colorbar(im)
    plt.title("Visualization of Data Count")
    plt.xlabel("Longitude Index")
    plt.ylabel("Latitude Index")
    plt.show()

    print("Processing complete. Data count shape:", final_data_count.shape)

if __name__ == '__main__':
    main()
