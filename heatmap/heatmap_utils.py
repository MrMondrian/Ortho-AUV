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
from sklearn.datasets import make_blobs



def calculate_coordinates_local(u, v, d, c_x, c_y, focal_x, focal_y):
     x_over_z = (c_x - u) / focal_x
     y_over_z = (c_y - v) / focal_y
     # this is in z north, y up, x east
     z = d / np.sqrt(1 + x_over_z**2 + y_over_z**2)
     x = x_over_z * z
     y = y_over_z * z 

     # convert to cam local frame
     x_cam_local = z
     y_cam_local = -x
     z_cam_local = y

     return x, y, z

def calculate_coordinates_world(local_x, local_y, local_z, auv_x, auv_y, auv_z, auv_quat, cam_position_offset):
     global_obj_pos_offset = quaternion.rotate_vectors(auv_quat, np.array([local_x, local_y, local_z]))
     global_x = global_obj_pos_offset[0] + auv_x + cam_position_offset[0]
     global_y = global_obj_pos_offset[1] + auv_y + cam_position_offset[1]
     global_z = global_obj_pos_offset[2] + auv_z + cam_position_offset[2]

     return global_x, global_y, global_z
     

def load_predictions2(path_to_predictions_file, path_to_depth_folder):
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

          x = int(float(x))
          y = int(float(y))
          width = int(float(width))
          height = int(float(height))

          predictions.append([path_to_depth_file, x, y, width, height])
     
     return predictions, repeat




































def normalize(depth_map):
    depth_map[np.isnan(depth_map)] = 0

    max_value = np.max(depth_map)
    min_value = np.min(depth_map)

    if max_value == min_value:
        return np.zeros_like(depth_map)

     # Normalize array to range [0, 255]
    normalized_depth = 255 * (depth_map - min_value) / (max_value - min_value)
    
    
    return normalized_depth.astype(np.uint8)


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


def calculate_and_plot_cluster(fish_positions, eps=15, min_samples=3):
     """ 
     creates and saves a 3D scatter plot with clusters colouring based on the proximity between fish positions

     args:
          fish_positions (list): list of position [x, y, z] of every fish detected
     
     returns:
          clusters_3d (np.ndarray): each element corresponds to the cluster that the i-th fish position belongs to (-1 = outliers)
     """
     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
     dbscan.fit(fish_positions)
     labels = dbscan.labels_

     # Create a scatter plot
     scatter = go.Scatter3d(
          x=fish_positions[:, 0],
          y=fish_positions[:, 1],
          z=fish_positions[:, 2],
          mode='markers',
          marker=dict(
               size=5,
               color=labels,
               colorscale='Viridis',
               opacity=0.8
          )
     )

     # Plot layout
     layout = go.Layout(
          title='3D Fish Positions Scatter Plot with DBSCAN Clusters',
          scene=dict(
               xaxis=dict(title='X'),
               yaxis=dict(title='Y'),
               zaxis=dict(title='Z')
          )
     )

     # Create the figure
     fig = go.Figure(data=[scatter], layout=layout)

     # Show the interactive plot
     fig.show()

     # Save the plot as HTML
     fig.write_html("3d_scatter_plot1.html", auto_open=True)


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


def make_heatmap(fish_positions):
     """
     returns a heatmap with fish distribution

     args:
          fish_positions (list): filtered list of the position [x, y, z] of every fish 
          auv_positions (list): list of all auv positions [x, y, z] 
     """ 
     kde = gaussian_kde(fish_positions.T)

     # Define grid for plotting
     x_grid, y_grid, z_grid = np.meshgrid(
          np.linspace(np.min(fish_positions[:, 0]), np.max(fish_positions[:, 0]), 50),
          np.linspace(np.min(fish_positions[:, 1]), np.max(fish_positions[:, 1]), 50),
          np.linspace(np.min(fish_positions[:, 2]), np.max(fish_positions[:, 2]), 50)
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
