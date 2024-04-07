import os
from PIL import Image
import numpy as np  # Assuming you want to save image data as NumPy arrays
import quaternion
import json
from argparse import ArgumentParser
import cv2

from heatmap_utils import *



predictions, repeat = load_predictions("vision/prediction.txt", "data4/")
poses = load_poses("data4/out_format.txt", repeat)

for i in range(len(predictions)):
    image_depth_path, x_0, x_1, y_0, y_1 = predictions[i]
    fish_dist = fish_distance(image_depth_path, x_0, x_1, y_0, y_1)
    predictions[i].append(fish_dist)

fish_positions = []

for i in range(len(predictions)):
    fish_dist = predictions[i][-1]
    camera_position = poses[i][:3]
    camera_orientation = poses[i][-1]

    fish_position = get_fish_position(fish_dist, camera_position, camera_orientation)

    fish_positions.append(fish_position)

# fish_positions = filter_fish_positions(fish_positions, ???position_time???)
# 
# fish_positions = np.array(fish_positions)

auv_positions = []

for pose in poses:
    auv_positions.append(pose[:3])
    
auv_positions = np.array(auv_positions)
fish_positions = np.array(fish_positions)

heatmap_size = get_heatmap_size(fish_positions, auv_positions) 

make_heatmap(fish_positions, heatmap_size)

""" 




"""



