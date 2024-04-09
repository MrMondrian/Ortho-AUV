import numpy as np 
import quaternion
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from pylab import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
from argparse import ArgumentParser


def load_camera_params(path_to_camera_params):
     """ 
     load camera params from json file

     args: 
          path_to_camera_params (string): path to camera params json file

     returns:
          width (int): width of camera fov 
          height (int): height of camera fov 
          focal_x (float): focal point x-coordinate
          focal_y (float): focal point y-coordinate  
          c_x (float): 
          c_y (float):
     """
     with open(path_to_camera_params, "r") as f:
          camera_data = json.load(f)

     # Extract the camera intrinsics parameters
     camera_params = list(camera_data.values())[0]
     width = camera_params['width']
     height = camera_params['height']
     focal_x = camera_params['focal_x']
     focal_y = camera_params['focal_y']
     c_x = camera_params['c_x']
     c_y = camera_params['c_y']

     return width, height, focal_x, focal_y, c_x, c_y


def camera_info(width, height, focal_x, focal_y, c_x, c_y):
     """ 
     calculate camera info

     args: 
          width (int): width of camera fov 
          height (int): height of camera fov 
          focal_x (float): focal point x-coordinate
          focal_y (float): focal point y-coordinate  
          c_x (float): 
          c_y (float):
     
     returns:
          x_over_z_map
          y_over_z_map
     """
     u_map = np.tile(np.arange(width), (height, 1)) + 1
     v_map = np.tile(np.arange(height), (width, 1)).T + 1

     x_over_z_map = (c_x - u_map) / focal_x
     y_over_z_map = (c_y - v_map) / focal_y

     return x_over_z_map, y_over_z_map


def load_predictions(path_to_predictions_file, path_to_depth_folder):
     """
     returns a list with all the info about the predictions info

     args:
          path_to_predictions (string): path to prediction text file

     return:
          predictions (list): [[path_to_depth_file, x_1, y_1, width_1, height_1], ... for all detection instances]
          repeat (dict): number of detected fish per image
     """
     with open(path_to_predictions_file, "r") as f:
          lines = f.readlines()
          lines = lines[1:]  # skip the header line
     
     predictions = []
     repeat = {}

     for line in lines:
          filename_raw, filename_depth, x, y, width, height = line.strip().split()
          
          if filename_raw in repeat:
               repeat[filename_raw] += 1
          else:
               repeat[filename_raw] = 1
          
          path_to_depth_file = path_to_depth_folder + filename_depth
          x = float(x)
          y = float(y)
          width = float(width)
          height = float(height)         

          # bounding box calculations:
          x_0 = int(x - width / 2)
          x_1 = int(x + width / 2)
          y_0 = int(y - height / 2)
          y_1 = int(y + height / 2)

          predictions.append([path_to_depth_file, x_0, x_1, y_0, y_1])
     
     return predictions, repeat


def load_poses(path_to_poses, repeat):
     """ 
     returns a list with all the auv poses for each prediction

     args:
          path_to_predictions (string): path to poses text file
          repeat (dict): number of detected fish per image

     return:
          poses (list): [[x_1, y_1, z_1, quat_1], ... for all instances]
     """
     with open(path_to_poses, "r") as f:
          lines = f.readlines()
          lines = lines[1:]  # skip the header line
     
     poses = []

     for line in lines:
          filename_raw, _, x, y, z, qw, qx, qy, qz = line.strip().split()

          n = repeat[filename_raw]

          # convert numerical data to appropriate types
          x = float(x)
          y = float(y)
          z = float(z)
          qw = float(qw)
          qx = float(qx)
          qy = float(qy)
          qz = float(qz)

          quat = np.quaternion(qw, qx, qy, qz)
          # quat = fix_quat(quat)

          for i in range(n):
               poses.append([x, y, z, quat])
     
     return poses


def get_xyz_image(z_map, width, height, x_over_z_map, y_over_z_map, cam_position_offset):
     cam_x_offset, cam_y_offset, cam_z_offset = cam_position_offset
     xyz_img = np.zeros((height, width, 3))

     x_map = x_over_z_map * -z_map
     y_map = y_over_z_map * -z_map

     xyz_img[:, :, 0] = y_map + cam_x_offset
     xyz_img[:, :, 1] = x_map + cam_y_offset
     xyz_img[:, :, 2] = -z_map + cam_z_offset

     return xyz_img


def crop_to_box(img, x_0, x_1, y_0, y_1):
     return img[y_0:y_1, x_0:x_1]


def clean_point_cloud(point_cloud):
     # Find the closest point to the camera
     initial_point_cloud_shape = point_cloud.shape
     point_cloud = point_cloud.reshape(-1,3)
     point_cloud[point_cloud[:,2] > -0.3] = -10000 # ignore depth values which are less than 0.5m
     closest_point_index = np.argmax(point_cloud[:,2])
     
     eps = 0.3  # Maximum distance between two samples for them to be considered as in the same neighborhood
     min_samples = 10  # The number of samples in a neighborhood for a point to be considered as a core point

     # Perform DBSCAN clustering
     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
     labels = dbscan.fit_predict(point_cloud)
     
     # Extract the cluster containing the seed point
     closest_object_label = labels[closest_point_index]
     object_mask = (labels == closest_object_label)
     point_cloud[~object_mask] = np.array([np.nan]*3)
     
     point_cloud = point_cloud.reshape(initial_point_cloud_shape)
     object_mask = object_mask.reshape(initial_point_cloud_shape[0:2])
     
     return point_cloud

def get_fish_position(point_cloud, auv_pose):
     auv_x, auv_y, auv_z, auv_quat = auv_pose

     min_lx = np.nanmin(point_cloud[:,:,0].flatten())
     min_ly = np.nanmin(point_cloud[:,:,1].flatten())
     min_lz = np.nanmin(point_cloud[:,:,2].flatten())
     max_lx = np.nanmax(point_cloud[:,:,0].flatten())
     max_ly = np.nanmax(point_cloud[:,:,1].flatten())
     max_lz = np.nanmax(point_cloud[:,:,2].flatten())
     
     lx = (max_lx + min_lx) / 2
     ly = (max_ly + min_ly) / 2
     lz = (max_lz + min_lz) / 2

     print(lx, ly, lz)
     # print(ly)
     # print(lz)
     
     global_obj_pos_offset = quaternion.rotate_vectors(auv_quat, np.array([lx,ly,lz]))
     
     # Get the best estimate of the mean
     x,y,z = global_obj_pos_offset + np.array([auv_x, auv_y, auv_z])

     return x, y, z

# def get_fish_position(prediction, distance_to_fish, camera_position, camera_orientation):
#      """
#      returns position of the fish

#      args:
#           prediction (list): [path_to_depth_file, x_1, y_1, width_1, height_1]
#           distance_to_fish (float): distance from the camera to the fish
#           camera_position list[float]: position of AUV [x, y, z]
#           camera_orientation (quaternion): orientation of camera
#      """
#      _, x, y, width, height = prediction

#      # convert quaternion angle to rotation matrix
#      rotation_matrix = quaternion.as_rotation_matrix(camera_orientation)
     
#      # initial vector pointing in the direction of the positive X-axis
#      initial_vector = np.array([1, 0, 0])
     
#      # rotate the initial vector using the rotation matrix
#      local_fish_vector = np.dot(rotation_matrix, initial_vector)
     
#      # scale the vector by the distance
#      local_fish_vector *= distance_to_fish

#      global_fish_vector = local_fish_vector + camera_position
     
#      return global_fish_vector
    

def filter_fish_positions(fish_positions):
     """
     returns a filtered/cleaned mapping of fish

     args:
          fish_positions (list): list of position [x, y, z] of every fish detected
     """
     pass


def calculate_and_plot_cluster(fish_positions):
     """ 
     creates and saves a 3D scatter plot with clusters colouring based on the proximity between fish positions

     args:
          fish_positions (list): list of position [x, y, z] of every fish detected
     
     returns:
          clusters_3d (np.ndarray): each element corresponds to the cluster that the i-th fish position belongs to (-1 = outliers)
     """
     # Defining and fitting the DBSCAN model
     dbscan = DBSCAN(eps=5, min_samples=2)
     clusters_3d = dbscan.fit_predict(fish_positions)

     # Extracting cluster labels and data points for each cluster
     unique_labels_3d = np.unique(clusters_3d)
     clusters_fish_positions = []
     for label in unique_labels_3d:
          class_member_mask_3d = (clusters_3d == label)
          xyz = fish_positions[class_member_mask_3d]
          clusters_fish_positions.append(xyz)

     # Plotting
     fig = go.Figure()

     # Add traces for each cluster
     for i, cluster_data in enumerate(clusters_fish_positions):
          if unique_labels_3d[i] == -1:
               name = 'Outliers'
     else:
          name = f'Cluster {unique_labels_3d[i]}'
     fig.add_trace(go.Scatter3d(
          x=cluster_data[:, 0],
          y=cluster_data[:, 1],
          z=cluster_data[:, 2],
          mode='markers',
          marker=dict(
               size=5,
               opacity=0.8,
          ),
          name=name
     ))

     # Update layout
     fig.update_layout(
     scene=dict(
          xaxis=dict(title='X-coordinate'),
          yaxis=dict(title='Y-coordinate'),
          zaxis=dict(title='Z-coordinate'),
     ),
     title='Fish Clustering'
     )

     # Save the plot as an HTML file
     fig.write_html('fish_clusters_3d_plot.html')

     return clusters_3d


def get_heatmap_size(fish_positions, auv_positions):
     """
     returns size of heatmap based on max and min fish and auv positions

     args:
          fish_positions (list): filtered list of the position [x, y, z] of every fish
          auv_positions (list): list of all auv positions [x, y, z]
     
     return:
          axis_size (list): [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
     """
     # @to-do fix tuple change 
     axis_size = []

     for i in range(3):
          min_axis = np.min(np.array(np.min(fish_positions[:, i]), np.min(auv_positions[:, i])))
          max_axis = np.max(np.array(np.max(fish_positions[:, i]), np.max(auv_positions[:, i])))
          axis_size.append((min_axis, max_axis))

     return axis_size


def make_heatmap(fish_positions, heatmap_size):
     """
     returns a heatmap with fish distribution

     args:
          fish_positions (list): filtered list of the position [x, y, z] of every fish 
          auv_positions (list): list of all auv positions [x, y, z] 
     """ 
     kde = gaussian_kde(fish_positions.T)

     # Define grid for plotting
     x_grid, y_grid, z_grid = np.meshgrid(
          np.linspace(fish_positions[:, 0].min(), fish_positions[:, 0].max(), 50),
          np.linspace(fish_positions[:, 1].min(), fish_positions[:, 1].max(), 50),
          np.linspace(fish_positions[:, 2].min(), fish_positions[:, 2].max(), 50)
     )

     density = kde(np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]))

     # Define threshold for transparency
     threshold = 0.2  # Adjust as needed

     # Normalize density to [0, 1]
     density_normalized = (density - density.min()) / (density.max() - density.min())

     # Set alpha based on density and threshold

     # Create scatter3d plot
     fig = go.Figure(data=go.Scatter3d(
          x=x_grid.ravel(),
          y=y_grid.ravel(),
          z=z_grid.ravel(),
          mode='markers',
          marker=dict(
               size=3,
               color=density_normalized,
               opacity=0.2,
               colorscale='viridis',
          )
     ))

     # Set layout
     fig.update_layout(
     scene=dict(
          xaxis_title='X-coordinate',
          yaxis_title='Y-coordinate',
          zaxis_title='Z-coordinate',
     ),
     title='Density Plot'
     )

     # Save plot as HTML file
     fig.write_html("density_plot_interactive.html")
     

# returns q_nwu_cam given q_nwu_auv
def fix_quat(q_nwu_auv):
     # quaternion that rotates the from the AUV to the camera
     q_auv_cam = np.quaternion(0, 0.707, -0.707, 0)

     return q_nwu_auv * q_auv_cam















# depth = 1
# arr = [1, 1, 1]
# quat_no_rotation = np.quaternion(1, 0, 0, 0)
# quat_45_roll = np.quaternion(0.9238795, 0.3826834, 0, 0)
# quat_90_roll = np.quaternion(0.7071068, 0.7071068, 0, 0)
# quat_180_roll = np.quaternion(0, 1, 0, 0)
# quat_45_pitch = np.quaternion(0.9238795, 0, 0.3826834, 0)
# quat_90_pitch = np.quaternion(0.7071068, 0, 0.7071068, 0)
# quat_180_pitch = np.quaternion(0, 0, 1, 0)
# quat_45_yaw = np.quaternion(0.9238795, 0, 0, 0.3826834)
# quat_90_yaw = np.quaternion(0.7071068, 0, 0, 0.7071068)
# quat_180_yaw = np.quaternion(0, 0, 0, 1)

# print(f"no rotation - {fish_position(depth, arr, quat_no_rotation)}")
# print(f"45 roll - {fish_position(depth, arr, quat_45_roll)}")
# print(f"90 roll - {fish_position(depth, arr, quat_90_roll)}")
# print(f"180 roll - {fish_position(depth, arr, quat_180_roll)}")
# print(f"45 pitch - {fish_position(depth, arr, quat_45_pitch)}")
# print(f"90 pitch - {fish_position(depth, arr, quat_90_pitch)}")
# print(f"180 pitch - {fish_position(depth, arr, quat_180_pitch)}")
# print(f"45 yaw - {fish_position(depth, arr, quat_45_yaw)}")
# print(f"90 yaw - {fish_position(depth, arr, quat_90_yaw)}")
# print(f"180 yaw - {fish_position(depth, arr, quat_180_yaw)}")

# quat_45_roll_45_yaw = np.quaternion(0.8535534, 0.3535534, -0.1464466, 0.3535534)
# quat_90_roll_90_pitch = np.quaternion(0.5, 0.5, 0.5, 0.5)
# quat_45_roll_45_pitch_45_yaw = np.quaternion(0.7325378, 0.4619398, 0.1913417, 0.4619398)

# print(f"quat_45_roll_45_yaw - {fish_position(depth, arr, quat_45_roll_45_yaw)}")
# print(f"quat_90_roll_90_pitch - {fish_position(depth, arr, quat_90_roll_90_pitch)}")
# print(f"quat_45_roll_45_pitch_45_yaw - {fish_position(depth, arr, quat_45_roll_45_pitch_45_yaw)}")







