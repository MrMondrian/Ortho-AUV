import os
from PIL import Image
import numpy as np  # Assuming you want to save image data as NumPy arrays
import quaternion
import json
from argparse import ArgumentParser

from heatmap_utils import *


# Define the quaternion that rotates the from the AUV to the camera
q_auv_cam = np.quaternion(0,0.707,-0.707,0)

# returns q_nwu_cam given q_nwu_auv
def fix_quat(q_nwu_auv):
    return q_nwu_auv * q_auv_cam


# Read data from "our_format.txt"
with open(os.path.join(data_dir, "our_format.txt"), "r") as f:
    lines = f.readlines()
    lines = lines[1:]  # Skip the header line