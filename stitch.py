import os
from PIL import Image
import numpy as np  # Assuming you want to save image data as NumPy arrays
import quaternion
import json


def pixel_coords(image_shape):
    """
    Returns a 2D NumPy array containing the normalized camera coordinates for each pixel
    in the given image shape.

    Args:
        image_shape (tuple): A tuple of two integers representing the image height and width.

    Returns:
        np.ndarray: A 2D NumPy array of shape (height, width, 2), where each element
                    contains the normalized camera coordinates (x, y) for the corresponding pixel.
    """
    height, width = image_shape

    # Create a grid of (x, y) coordinates in the range [-1, 1]
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(1, -1, height))

    # Stack the x and y coordinates into a 2D array of shape (height, width, 2)
    coords = np.dstack((x, y))

    return coords


def apply_matrix_to_vectors(array_3d, transformation_matrix):
    """
    Applies a 3x3 transformation matrix to each vector3 in a 3D NumPy array.
    
    Args:
        array_3d (np.ndarray): A 3D NumPy array where each element is a vector3.
        transformation_matrix (np.ndarray): A 3x3 transformation matrix.
        
    Returns:
        np.ndarray: A 3D NumPy array with the same shape as the input array, where each vector3 has been transformed by the given matrix.
    """
    # Ensure the input array is a 3D NumPy array
    array_3d = np.asarray(array_3d)
    if array_3d.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array.")
    
    # Ensure the transformation matrix is a 3x3 NumPy array
    transformation_matrix = np.asarray(transformation_matrix)
    if transformation_matrix.shape != (3, 3):
        raise ValueError("Transformation matrix must be a 3x3 NumPy array.")
    
    # Reshape the input array to a 2D array of vectors
    vectors = array_3d.reshape(-1, 3)
    
    # Apply the transformation matrix to each vector
    transformed_vectors = np.dot(vectors, transformation_matrix.T)
    
    # Reshape the transformed vectors back to the original 3D shape
    transformed_array = transformed_vectors.reshape(array_3d.shape)
    
    return transformed_array


def calculate_H_matrix(image_data_a, image_data_b):
    quat_a = image_data_a[5]
    quat_b = image_data_b[5]

    d = image_data_b[3]

    n = np.array([0,0,1])

    ta = np.array(image_data_a[1:4])
    tb = np.array(image_data_b[1:4])

    Ra = quaternion.as_rotation_matrix(quat_a)
    RbT = np.transpose(quaternion.as_rotation_matrix(quat_b))

    Rabt = np.matmul(Ra,RbT)

    H = -np.matmul(Rabt,tb) + ta
    H = np.matmul(H,n.transpose()) / d
    H = Rabt - H
    return H

def calculate_transformation_matrix(image_data_a, image_data_b):
    Hab = calculate_H_matrix(image_data_a,image_data_b)
    # for our data set the zs are equal
    return K @ Hab @ Kt

def transform_image(src, target):
    mat = calculate_transformation_matrix(target, src)
    src[-1] = apply_matrix_to_vectors(mat,src[1])



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
            images.append((filename, image_data, x, y, z, quat, pixel_coords(image_data.shape)))
    except Exception as e:
        print(f"Error processing image {filename}: {e}")
