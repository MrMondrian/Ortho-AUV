import numpy as np  # Assuming you want to save image data as NumPy arrays
import quaternion
from scipy.ndimage import gaussian_filter
import cv2
from heatmap_utils import *


width, height, focal_x, focal_y, c_x, c_y = load_camera_params("data/cameras.json")
x_over_z_map, y_over_z_map = camera_info(width, height, focal_x, focal_y, c_x, c_y)
cam_position_offset = [0.2, 0.1, -0.2]

predictions, repeat = load_predictions("vision/prediction.txt", "data/")
poses = load_poses("data/out_format.txt", repeat)
fish_positions = []

def normalize(depth_map):
    depth_map[np.isnan(depth_map)] = 0

    max_value = np.max(depth_map)
    min_value = np.min(depth_map)

    if max_value == min_value:
        return np.zeros_like(depth_map)

# Normalize array to range [0, 255]
    normalized_depth = 255 * (depth_map - min_value) / (max_value - min_value)
    
    
    return normalized_depth.astype(np.uint8)



for i in range(len(predictions)):
    image_depth_path, x_0, x_1, y_0, y_1 = predictions[i]
    z_map = np.loadtxt(image_depth_path)
    point_cloud = get_xyz_image(z_map, width, height, x_over_z_map, y_over_z_map, cam_position_offset)
    # print(np.min(point_cloud[:,:,2]))
    # print(np.max(point_cloud))
    point_cloud = crop_to_box(point_cloud, x_0, x_1, y_0, y_1)
    # print(point_cloud)
    point_cloud = clean_point_cloud(point_cloud)

    normalized_depth = normalize(point_cloud)
    cv2.imwrite(f"normalized_image{i}.png", normalized_depth)
    x, y, z = get_fish_position(point_cloud, poses[i])
    fish_positions.append([x, y, z])

print(fish_positions)
auv_positions = []

for pose in poses:
    auv_positions.append(pose[:3])
    
fish_positions = np.array(fish_positions)

calculate_and_plot_cluster(fish_positions)





