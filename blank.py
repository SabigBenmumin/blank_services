import laspy
import numpy as np
import open3d as o3d
import math
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
import tqdm
import redis

def read_last_file(file_path):
    las = laspy.read(file_path)
    points = np.vstack([las.x, las.y, las.z]).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return las, pcd

def las2Array(lasdata):
    points = np.vstack([lasdata.x, lasdata.y, lasdata.z]).transpose()

    if hasattr(lasdata, 'red') and hasattr(lasdata, 'green') and hasattr(lasdata, 'blue'):
        colors = np.vstack([lasdata.red, lasdata.green, lasdata.blue]).transpose()
        colors = colors / 65535.0
    else:
        colors = np.ones((len(points), 3))
    return points, colors

def get_pcd(file_path):
    las_data, pcd = read_last_file(file_path)
    points, colors = las2Array(las_data)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def calculate_grid(points, x_min, y_min, col_width, row_width, n_rows, n_cols):
    kdtree = cKDTree(points[:, :2])

    avg_alt_mat = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    point_counts = np.zeros((n_rows, n_cols), dtype=int)
    for row in tqdm.tqdm(range(n_rows)):
        for col in range(n_cols):

            x0 = x_min + col * col_width
            x1 = x0 + col_width
            y0 = y_min + row * row_width
            y1 = y0 + row_width

            cell_center = [(x0 + x1) / 2, (y0 + y1) / 2]
            max_dist = math.sqrt((col_width / 2) ** 2 + (row_width / 2) ** 2)
            cell_indices = kdtree.query_ball_point(cell_center, max_dist)

            if len(cell_indices) == 0:
                continue

            cell_points = points[cell_indices]
            cell_points = cell_points[
                (cell_points[:, 0] >= x0) & (cell_points[:, 0] < x1) &
                (cell_points[:, 1] >= y0) & (cell_points[:, 1] < y1)
            ]

            if cell_points.size > 0:
                avg_alt_mat[row, col] = np.median(cell_points[:, 2])
                point_counts[row, col] = cell_points.shape[0]

    return avg_alt_mat, point_counts

def plot_graph(src_avg_alt_mat, tgt_avg_alt_mat, delta_alt_mat, gbl_z_min, gbl_z_max, width_meters, height_meters, sand_increase, sand_decrease, task_id, result_name='result', dpi=300):
    norm = Normalize(vmin=gbl_z_min, vmax=gbl_z_max)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[1, 1].axis('off')
    # Subplot 1: Average Altitude (Source)
    im1 = axes[0, 0].imshow(src_avg_alt_mat, cmap='viridis_r', norm=norm)
    axes[0, 0].set_title('Average Altitude (Time Series1)')
    axes[0, 0].set_xlabel(f'Width {width_meters:.2f} meters')
    axes[0, 0].set_ylabel(f'Height {height_meters:.2f} meters')
    axes[0, 0].invert_yaxis()

# Subplot 2: Average Altitude (Target)
    im2 = axes[0, 1].imshow(tgt_avg_alt_mat, cmap='viridis_r', norm=norm)
    axes[0, 1].set_title('Average Altitude (Time Series2)')
    axes[0, 1].set_xlabel(f'Width {width_meters:.2f} meters')
    axes[0, 1].set_ylabel(f'Height {height_meters:.2f} meters')
    axes[0, 1].invert_yaxis()

# Subplot 3: Delta Altitude (Volume Change)
    im3 = axes[1, 0].imshow(delta_alt_mat, cmap='viridis_r', norm=norm)
    axes[1, 0].set_title('Delta Altitude (Volume Change)')
    axes[1, 0].set_xlabel(f'Width {width_meters:.2f} meters')
    axes[1, 0].set_ylabel(f'Height {height_meters:.2f} meters')
    axes[1, 0].invert_yaxis()

# Subplot 4: Period of change
    threshold_min = -0.1
    threshold_max = 0.1
    norm1 = Normalize(vmin=threshold_min, vmax=threshold_max)
    im4 = axes[1, 1].imshow(delta_alt_mat, cmap='bwr', norm=norm1)
    axes[1, 1].set_title('Period of change')
    axes[1, 1].set_xlabel('Width (meters)')
    axes[1, 1].set_ylabel('Height (meters)')
    axes[1, 1].text(0.5, -0.15, f'> 0.1 m : Sand Increase(Red)\n< -0.1 m : Sand Decrease(Blue)\n Sand Volume Increase: {sand_increase:.2f} m³\nSand Volume Decrease: {sand_decrease:.2f} m³',
        fontsize=12, ha='center', va='top', transform=axes[1, 1].transAxes)
    axes[1, 1].invert_yaxis()

# Add colorbars for each graph
    fig.colorbar(im1, ax=axes[0, 0])
    fig.colorbar(im2, ax=axes[0, 1])
    fig.colorbar(im3, ax=axes[1, 0])
    fig.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.title("Volume Change Analysis")
    result_path = f'/mnt/d/blankspace/blankservices/result_store/{result_name}_{task_id}.png'
    # result_path_windows = f'D:/blankspace/blankservices/result_store/{result_name}_{task_id}.png'

    plt.savefig(result_path, dpi=dpi)

    
    # return result_path_windows

def calculator(source_path, target_path, task_id, grid_size=0.1):
    src_pcd = get_pcd(source_path)
    tgt_pcd = get_pcd(target_path)

    src_points = np.asarray(src_pcd.points)
    tgt_points = np.asarray(tgt_pcd.points)

    #Bounding box and grid setup
    src_x_min, src_x_max = min(src_points[:, 0]), max(src_points[:, 0])
    src_y_min, src_y_max = min(src_points[:, 1]), max(src_points[:, 1])
    src_z_min, src_z_max = min(src_points[:, 2]), max(src_points[:, 2])
    tgt_x_min, tgt_x_max = min(tgt_points[:, 0]), max(tgt_points[:, 0])
    tgt_y_min, tgt_y_max = min(tgt_points[:, 1]), max(tgt_points[:, 1])
    tgt_z_min, tgt_z_max = min(tgt_points[:, 2]), max(tgt_points[:, 2])

    gbl_x_min, gbl_x_max = min(src_x_min, tgt_x_min), max(src_x_max, tgt_x_max)
    gbl_y_min, gbl_y_max = min(src_y_min, tgt_y_min), max(src_y_max, tgt_y_max)

    COL_WIDTH = grid_size
    ROW_WIDTH = grid_size

    n_cols = math.ceil((gbl_x_max - gbl_x_min) / COL_WIDTH)
    n_rows = math.ceil((gbl_y_max - gbl_y_min) / ROW_WIDTH)

    # Prepare argument tuple for multiprocessing
    src_args = (src_points, gbl_x_min, gbl_y_min, COL_WIDTH, ROW_WIDTH, n_rows, n_cols)
    tgt_args = (tgt_points, gbl_x_min, gbl_y_min, COL_WIDTH, ROW_WIDTH, n_rows, n_cols)

    if n_cols * n_rows < 1:
        with multiprocessing.Pool(processes=2) as pool:
            results = pool.starmap(calculate_grid, [src_args, tgt_args])
        
        # Unpack results
        src_avg_alt_mat, src_point_counts = results[0]
        tgt_avg_alt_mat, tgt_point_counts = results[1]
    else:
        src_avg_alt_mat, src_point_counts = calculate_grid(src_points, gbl_x_min, gbl_y_min, COL_WIDTH, ROW_WIDTH, n_rows, n_cols)
        tgt_avg_alt_mat, tgt_point_counts = calculate_grid(tgt_points, gbl_x_min, gbl_y_min, COL_WIDTH, ROW_WIDTH, n_rows, n_cols)
    
    delta_alt_mat = tgt_avg_alt_mat - src_avg_alt_mat

    cell_area = COL_WIDTH * ROW_WIDTH
    # volume_change = (np.array(tgt_avg_alt_mat) - np.array(src_avg_alt_mat)) * cell_area
    volume_change = (delta_alt_mat) * cell_area

    volume_change = np.nan_to_num(volume_change)
    total_volume_change = np.sum(volume_change)

    gbl_z_min = min(src_z_min, tgt_z_min)
    gbl_z_max = max(src_z_max, tgt_z_max)
    width_meters = gbl_x_max - gbl_x_min
    height_meters = gbl_y_max - gbl_y_min
    sand_increase = np.sum(delta_alt_mat[delta_alt_mat > 0.1] * cell_area)
    sand_decrease = np.sum(delta_alt_mat[delta_alt_mat < -0.1] * cell_area)
    result_path = plot_graph(src_avg_alt_mat, tgt_avg_alt_mat, delta_alt_mat, gbl_z_min, gbl_z_max, width_meters, height_meters, sand_increase, sand_decrease, task_id)

    return total_volume_change, sand_increase, sand_decrease