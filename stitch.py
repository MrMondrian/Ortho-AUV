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
    height, width, _ = image_shape

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

    ta = np.array([image_data_a[2],image_data_a[4],image_data_a[4]])
    tb = np.array([image_data_b[2],image_data_b[4],image_data_b[4]])

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

def get_img_bound(image_data):
    coords = image_data[6]
    corner_x_values = [coords[0][0][0], coords[0][-1][0], coords[-1][0][0], coords[-1][-1][0]]
    corner_y_values = [coords[0][0][1], coords[0][-1][1], coords[-1][0][1], coords[-1][-1][1]]
    return np.array([min(corner_x_values), max(corner_x_values), min(corner_y_values), max(corner_y_values)])

def get_bounds_union(bounds):
    min_x = np.min(bounds[:,0])
    max_x = np.max(bounds[:,1])
    min_y = np.min(bounds[:,2])
    max_y = np.max(bounds[:,3])
    return np.array([min_x, max_x, min_y, max_y])

def get_image_dimesion(union_bounds,width,height):
    # takes in bound in normalized camera coordinates, returns the size of the image
    min_x = union_bounds[0]
    max_x = union_bounds[1]
    min_y = union_bounds[2]
    max_y = union_bounds[3]
    return (int((max_x - min_x) * width), int((max_y - min_y) * height))

def transform_image(src, target):
    width, height, _ = src[1].shape
    mat = calculate_transformation_matrix(target, src)
    src[-1] = apply_matrix_to_vectors(src[1],mat)
    src_bounds = get_img_bound(src)
    target_bounds = get_img_bound(target)
    union_bounds = get_bounds_union(np.array([src_bounds, target_bounds]))
    new_dim = get_image_dimesion(union_bounds, width, height)
    new_img = np.zeros(new_dim)
    ndc_to_vp = get_ndc_to_vp_matrix(union_bounds, width, height)
    for i in range(new_dim[0]):
        for j in range(new_dim[1]):
            ndc = np.array([i,j,1])
            vp = np.matmul(ndc_to_vp, ndc)
            if vp[0] >= 0 and vp[0] < width and vp[1] >= 0 and vp[1] < height:
                new_img[i,j] = src[1][vp[0],vp[1]]
    return new_img


def get_ndc_to_vp_matrix(union_bounds, width, height):
    min_x = union_bounds[0]
    max_y = union_bounds[3]
    return np.array([[width/2, 0, -min_x * width/2],
                     [0, -height/2, -max_y * height/2],
                     [0, 0, 1]])



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
    quat_to_downcam = np.quaternion(0.707,0,0.707,0)
    quat = quat * quat_to_downcam

    # Process the image data (assuming you have the code from previous part)
    filepath = os.path.join(data_dir, filename)
    try:
        with Image.open(filepath) as image:
            image_data = np.array(image)
            images.append([filename, np.array(image_data), x, y, z, quat, pixel_coords(image_data.shape)])
    except Exception as e:
        print(f"Error processing image {filename}: {e}")

# Stitch the images together
stitch1 = transform_image(images[0], images[1])
print(stitch1.shape)
# Save the stitched image
stitch1_image = Image.fromarray(stitch1).convert("RGB")
stitch1_image.save("stitch1.png")