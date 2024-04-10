import numpy as np  # Assuming you want to save image data as NumPy arrays
import quaternion
from scipy.ndimage import gaussian_filter
import cv2
from heatmap_utils import *


width, height, focal_x, focal_y, c_x, c_y = load_camera_params("data/cameras.json")
# x_over_z_map, y_over_z_map = camera_info(width, height, focal_x, focal_y, c_x, c_y)
cam_position_offset = [0.2, 0.1, -0.2]

# predictions, repeat = load_predictions("vision/prediction.txt", "data/")
predictions, repeat = load_predictions2("vision/prediction.txt", "data/")
poses = load_poses("data/out_format.txt", repeat)
fish_positions = []

for i in range(len(predictions)):
    image_depth_path, x, y, width, height = predictions[i]
    auv_x, auv_y, auv_z, auv_quat = poses[i]
    u, v = x, y
    d = np.loadtxt(image_depth_path)[v, u]
    local_x, local_y, local_z = calculate_coordinates_local(u, v, d, c_x, c_y, focal_x, focal_y)
    global_x, global_y, global_z = calculate_coordinates_world(local_x, local_y, local_z, auv_x, auv_y, auv_z, auv_quat, cam_position_offset)
    for i in range(10):
        fish_positions.append([global_x + i/10, global_y + i/10, global_z + i/10])
        fish_positions.append([global_x - i/10, global_y - i/10, global_z - i/10])


    
# for i in range(len(predictions)):
#     image_depth_path, x_0, x_1, y_0, y_1 = predictions[i]
#     z_map = np.loadtxt(image_depth_path)
#     point_cloud = get_xyz_image(z_map, width, height, x_over_z_map, y_over_z_map, cam_position_offset)
#     point_cloud = crop_to_box(point_cloud, x_0, x_1, y_0, y_1)
#     point_cloud = clean_point_cloud(point_cloud)

#     normalized_depth = normalize(point_cloud)
#     cv2.imwrite(f"normalized_image{i}.png", normalized_depth)
#     x, y, z = get_fish_position(point_cloud, poses[i])
#     fish_positions.append([x, y, z])

# print(fish_positions)
# auv_positions = []

# for pose in poses:
#     auv_positions.append(pose[:3])
    
# fish_positions = np.array(fish_positions)


# poses = np.array(poses)
# axis_size = get_heatmap_size(fish_positions, poses)
poses = np.random.rand(50, 3) 
fish_positions = np.random.rand(50, 3) 

for i in range(50):
    if i % 2 == 0:
        poses[i] *= 50
        fish_positions[i] *= 50
    else: 
        poses[i] *= -50
        fish_positions[i] *= -50

calculate_and_plot_cluster(fish_positions)
# make_heatmap(fish_positions)





