import numpy as np 
import quaternion

def fish_distance(img_depth):
     """ 
     returns the distance from the camera to the fish 

     args:
          img (np.array): a 2D array representing the depth camera values after fish detection on raw image   
     """
     # Find min value of the image and get distance points within 1 std_dev of min_value
     min_value = np.min(img_depth)
     std_dev = np.std(img_depth)
     min_range = min_value + std_dev
     max_range = min_value - std_dev

     within_range = img_depth[(img_depth >= max_range) & (img_depth <= min_range)]

     # distance to the fish will be average of the values within 1 std_dev
     return np.average(within_range)

def fish_position(distance_to_fish, camera_position, camera_orientation):
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
          fish_positions (list): list of position (x, y, z) of every fish detected
          position_time (list): the time at which each position was calculated
     """
     pass

def make_heatmap(fish_positions):
     """
     returns a heatmap with fish distribution

     args:
          fish_positions (list): filtered list of the position (x, y, z) of every fish 
     """
     pass

def save_heatmap(heatmap, path):
     """
     saves the heatmap in the given path

     args:
          heatmap (???): heatmap representation of fish distribution
          path (string): path to save the heatmap
     """
     pass


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


