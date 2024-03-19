import os
from PIL import Image
import numpy as np  # Assuming you want to save image data as NumPy arrays
import quaternion

data_dir = "data"  # Path to the data directory
images = []  # List to store image data and corresponding information

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



def calculate_homo_matrix(image_data_a, image_data_b):
    quat_a = image_data_a[5]
    quat_b = image_data_b[5]

    xyz_a = np.array(image_data_a[1:4])
    xyz_b = np.array(image_data_b[1:4])

    forward_quat = np.quaternion(0,1,0,0)
    norm_a_quat = quat_a * forward_quat * quat_a.conjugate()
    norm_b_quat = quat_b * forward_quat * quat_b.conjugate()

    norm_a = np.array([norm_a_quat.x,norm_a_quat.y,norm_a_quat.z])
    norm_b = np.array([norm_b_quat.x,norm_b_quat.y,norm_b_quat.z])

    xyz_a_to_b = xyz_b - xyz_a

    Ra = quaternion.as_rotation_matrix(quat_a)
    RbT = quaternion.as_rootation_matrix(quat_b).transpose()
    


