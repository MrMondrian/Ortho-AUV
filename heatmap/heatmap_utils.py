import numpy as np 
import quaternion
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN
from pylab import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def load_predictions(path_to_predictions_file, path_to_depth_folder):
     """
     returns a list with all the info about the predictions info

     args:
          path_to_predictions (string): path to prediction text file

     return:
          predictions (list): [[x_1, y_1, width_1, height_1], ... for all instances]
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
          quat = fix_quat(quat)

          for i in range(n):
               poses.append([x, y, z, quat])
     
     return poses

def fish_distance(image_depth_path, x_0, x_1, y_0, y_1):
     """ NOT TESTED - ARRAY SLICING MIGHT BE WRONG """
     """ 
     returns the distance from the camera to the fish 

     args:
          img (float): path to image depth 
          x_0, x_1, y_0, x_1 (int, int, int, int): bounding box coordinates
     """
     # load depth map
     depth_map = np.loadtxt(image_depth_path)

     # select only bounding box
     fish_depth_cut = depth_map[y_0 : y_1 + 1, x_0 : x_1 + 1]

     # Find min value of the image and get distance points within 1 std_dev of min_value
     min_value = np.min(fish_depth_cut)
     std_dev = np.std(fish_depth_cut)
     min_range = min_value + std_dev
     max_range = min_value - std_dev

     within_range = fish_depth_cut[(fish_depth_cut >= max_range) & (fish_depth_cut <= min_range)]

     # distance to the fish will be average of the values within 1 std_dev
     return np.average(within_range)

def get_fish_position(distance_to_fish, camera_position, camera_orientation):
     """
     returns position of the fish

     args:
          distance_to_fish (float): distance from the camera to the fish
          camera_position list[float]: position of AUV [x, y, z]
          camera_orientation (quaternion): orientation of camera
     """
     # convert quaternion angle to rotation matrix
     rotation_matrix = quaternion.as_rotation_matrix(camera_orientation)
     
     # initial vector pointing in the direction of the positive X-axis
     initial_vector = np.array([1, 0, 0])
     
     # rotate the initial vector using the rotation matrix
     local_fish_vector = np.dot(rotation_matrix, initial_vector)
     
     # scale the vector by the distance
     local_fish_vector *= distance_to_fish

     global_fish_vector = local_fish_vector + camera_position
     
     return global_fish_vector
    

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
     kde = gaussian_kde(data.T)

     # Define grid for plotting
     x_grid, y_grid, z_grid = np.meshgrid(
          np.linspace(data[:, 0].min(), data[:, 0].max(), 50),
          np.linspace(data[:, 1].min(), data[:, 1].max(), 50),
          np.linspace(data[:, 2].min(), data[:, 2].max(), 50)
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







