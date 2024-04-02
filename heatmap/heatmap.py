import os
from PIL import Image
import numpy as np  # Assuming you want to save image data as NumPy arrays
import quaternion
import json
from argparse import ArgumentParser

from heatmap_utils import *



predictions = load_predictions("../vision/prediction.txt")
poses = load_poses("../data/out_format.txt")

""" 
for i in range(len(predictions)):
    add point_cloud values
    distance = fish_distance(point_cloud)
    predictions[i].append(distance)

fish_positions = []

for i in range(len(predictions)):
    distance_to_fish = predictions[i][-1]
    camera_position = poses[i][:3]
    camera_orientation = poses[i][-1]
    fish_position = fish_position(distance_to_fish, camera_position, camera_orientation)
    fish_positions.append(fish_position)

fish_positions = filter_fish_positions(fish_positions, ???position_time???)

fish_positions = np.array(fish_positions)
auv_positions = np.array(poses[:][:3])

heatmap_size = get_heatmap_size(fish_positions, auv_positions) 

make_heatmap(fish_positions, heatmap_size)

"""



