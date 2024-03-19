import os
from PIL import Image
import numpy as np  # Assuming you want to save image data as NumPy arrays
import quaternion
import json

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

Kt = np.transpose(K)

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

    # Process the image data (assuming you have the code from previous part)
    filepath = os.path.join(data_dir, filename)
    try:
        with Image.open(filepath) as image:
            image_data = np.array(image)
            images.append((filename, image_data, x, y, z, quat))
    except Exception as e:
        print(f"Error processing image {filename}: {e}")



def calculate_H_matrix(image_data_a, image_data_b):
    quat_a = image_data_a[5]
    quat_b = image_data_b[5]

    d = image_data_b[3]

    n = np.array([0,0,1])

    ta = np.array(image_data_a[1:4])
    tb = np.array(image_data_b[1:4])

    Ra = quaternion.as_rotation_matrix(quat_a)
    RbT = quaternion.as_rotation_matrix(quat_b).transpose()

    Rabt = np.matmul(Ra,RbT)

    H = -np.matmul(Rabt,tb) + ta
    H = np.matmul(H,n.transpose()) / d
    H = Rabt - H
    return H

def calculate_transformation_matrix(image_data_a, image_data_b):
    Hab = calculate_H_matrix(image_data_a,image_data_b)
    # for our data set the zs are equal
    return K @ Hab @ Kt
    


