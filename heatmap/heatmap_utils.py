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

def fish_position(depth, x, y, z, quat):
     """
     returns translation of the fish

     args:
          depth (float): distance from the AUV to the fish
          x, y, z (float): translation of AUV
          quat (quaternion): orientation of AUV
     """
     euler = quaternion.as_euler(quat)
     roll, pitch = euler[0], euler[1]
     fish_local_vec = []
     fish_local_vec[0] = np.cos(roll)
     fish_local_vec[1] = np.cos(pitch)
     fish_local_vec[2] = depth
     
     fish_global = quaternion.rotate(quat, fish_local_vec)

     return fish_global
      
     

def filter_fish_translations(fish_translations, translation_time):
     """
     returns a filtered/cleaned mapping of fish

     args:
          fish_translations (list): list of translation (x, y, z) of every fish detected
          translation_time (list): the time at which each translation was calculated
     """
     pass

def make_heatmap(fish_translations):
     """
     returns a heatmap with fish distribution

     args:
          fish_translations (list): filtered list of the translation (x, y, z) of every fish 
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