import cv2 
import numpy as np 

def load_depth(depth_file):
     return np.loadtxt(depth_file)
          
def normalize(depth_map):
     max_value = np.max(depth_map)
     min_value = np.min(depth_map)

     if max_value == min_value:
          return np.zeros_like(depth_map)
    
    # Normalize array to range [0, 255]
     normalized_depth = 255 * (depth_map - min_value) / (max_value - min_value)
     
     
     return normalized_depth.astype(np.uint8)

depth_file = "data4/22:28:23.171_depth.txt"
depth_map = load_depth(depth_file)
normalized_depth = normalize(depth_map)
cv2.imwrite("normalized_image.png", normalized_depth)
