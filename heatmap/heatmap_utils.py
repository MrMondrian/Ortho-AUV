import numpy as np 
import quaternion

def fish_distance(img_depth):
     """ 
     returns the distance from the AUV to the fish

     args:
          img (np.array): a 2D array representing the depth camera values after fish detection on raw image   
     """
     pass

def fish_position(depth, x, y, z, quat):
     """
     returns translation of the fish

     args:
          depth (float): distance from the AUV to the fish
          x, y, z (float): translation of AUV
          quat (quaternion): orientation of AUV
     """
     pass

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