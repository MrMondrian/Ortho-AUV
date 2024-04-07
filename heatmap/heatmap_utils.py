import numpy as np 
import quaternion
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
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
    

def filter_fish_positions(fish_positions, position_time):
     """
     returns a filtered/cleaned mapping of fish

     args:
          fish_positions (list): list of position [x, y, z] of every fish detected
          position_time (list): the time at which each position was calculated
     """
     pass

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
     # creating figures 
     fig = plt.figure(figsize=(10, 10)) 
     ax = fig.add_subplot(111, projection='3d') 

     # set axis to be the size of observable lake (auv positions + fish positions)
     ax.set_xlim(heatmap_size[0][0], heatmap_size[0][1])
     ax.set_ylim(heatmap_size[1][0], heatmap_size[1][1])
     ax.set_zlim(heatmap_size[2][0], heatmap_size[2][1])
     
     x = fish_positions[:, 0]
     y = fish_positions[:, 1]
     z = fish_positions[:, 2]

     # creating the heatmap 
     img = ax.scatter(x, y, z, s=10, color='coral') 

     # adding title and labels 
     ax.set_title("3D Fish Positions Heatmap") 
     ax.set_xlabel('X-coordinate') 
     ax.set_ylabel('Y-coordinate') 
     ax.set_zlabel('Z-coordinate') 

     # convert plot to html 3d interactive plot
     plotly_fig = go.Figure()
     plotly_fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='green')))

     plotly_fig.write_html("interactive_heatmap.html")




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

# import numpy as np
# from scipy import stats
# from mayavi import mlab

# mu, sigma = 0, 0.1 
# x = 10*np.random.normal(mu, sigma, 500)
# y = 10*np.random.normal(mu, sigma, 500)    
# z = 10*np.random.normal(mu, sigma, 500)

# xyz = np.vstack([x,y,z])
# kde = stats.gaussian_kde(xyz)

# # Evaluate kde on a grid
# xmin, ymin, zmin = x.min(), y.min(), z.min()
# xmax, ymax, zmax = x.max(), y.max(), z.max()
# xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
# coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
# density = kde(coords).reshape(xi.shape)

# # Plot scatter with mayavi
# figure = mlab.figure('DensityPlot')

# grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
# min = density.min()
# max=density.max()
# mlab.pipeline.volume(grid, vmin=min, vmax=min + .5*(max-min))

# mlab.axes()
# mlab.show()

# fig = plt.figure(figsize=(10, 10)) 
# ax = fig.add_subplot(111, projection='3d') 

# x = np.random.rand(100)
# y = np.random.rand(100)
# z = np.random.rand(100)

# # creating the heatmap 
# img = ax.scatter(x, y, z, s=10, color='coral') 

# # adding title and labels 
# ax.set_title("3D Fish Positions Heatmap") 
# ax.set_xlabel('X-coordinate') 
# ax.set_ylabel('Y-coordinate') 
# ax.set_zlabel('Z-coordinate') 

# # convert plot to html 3d interactive plot
# plotly_fig = go.Figure()
# plotly_fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='green')))

# plotly_fig.write_html("interactive_heatmap.html")