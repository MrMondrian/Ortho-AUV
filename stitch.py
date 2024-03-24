import os
from PIL import Image
import numpy as np  # Assuming you want to save image data as NumPy arrays
import quaternion
import json
from argparse import ArgumentParser

from stitch_utils import *



# Define the quaternion that rotates the from the AUV to the camera
q_auv_cam = np.quaternion(0,0.707,-0.707,0)

# returns q_nwu_cam given q_nwu_auv
def fix_quat(q_nwu_auv):
    return q_nwu_auv * q_auv_cam

# get data dir with argument parser
parser = ArgumentParser()
parser.add_argument("data_dir", type=str, default="data")
args = parser.parse_args()

data_dir = args.data_dir

data_dir = "data"  # Path to the data directory
images = []  # List to store image data and corresponding information

# Load the JSON data
with open(os.path.join(data_dir, "cameras.json"), "r") as f:
    camera_data = json.load(f)

# Extract the camera intrinsics parameters
camera_params = list(camera_data.values())[0]
width = camera_params['width']
height = camera_params['height']
focal_x = camera_params['focal_x']
focal_y = camera_params['focal_y']
c_x = camera_params['c_x']
c_y = camera_params['c_y']

# Create the camera intrinsics matrix
K = np.array([[focal_x, 0, c_x],
              [0, focal_y, c_y],
              [0, 0, 1]])


# Read data from "our_format.txt"
with open(os.path.join(data_dir, "our_format.txt"), "r") as f:
    lines = f.readlines()
    lines = lines[1:]  # Skip the header line

# Iterate through lines in "our_format.txt"
for line in lines:
    filename, x, y, z, qw, qx, qy, qz = line.strip().split()

    # Convert numerical data to appropriate types
    x = float(x)
    y = float(y)
    z = float(z)
    qw = float(qw)
    qx = float(qx)
    qy = float(qy)
    qz = float(qz)

    quat = np.quaternion(qw,qx,qy,qz)
    quat = fix_quat(quat)


    # Process the image data (assuming you have the code from previous part)
    filepath = os.path.join(data_dir, filename)
    try:
        with Image.open(filepath) as image:
            image_data = np.array(image)
            img_dict = {"filename": filename, "image_data": image_data, "position": np.array([x,y,z]), "quat": quat, "pixel_coords": pixel_coords(image_data.shape), "intrinsics": K}
            images.append(img_dict)
    except Exception as e:
        print(f"Error processing image {filename}: {e}")


index = -1#len(images) // 2
constant = images[index]
images.pop(index)
i = 1
for image_data in images:
    print(i)
    i+=1
    transform_image(image_data, constant)

stitched_image = render_image(constant, images)
stitched_image = Image.fromarray(stitched_image).convert("RGB")
stitched_image.save("stitched_image.png")
