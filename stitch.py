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

    ratio = height / width

    # Create a grid of (x, y) coordinates in the range [-1, 1]
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-ratio, ratio, height))

    # Stack the x and y coordinates into a 2D array of shape (height, width, 2)
    coords = np.dstack((x, y,np.ones_like(x)))

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
        # out = np.matmul(np.matmul(K, Hab),np.linalg.inv(K))

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

    # divide by homogenous coordinate
    transformed_array = transformed_array / transformed_array[:,:,2].reshape(array_3d.shape[0],array_3d.shape[1],1)
    
    return transformed_array

# def nwu_to_neu_quat(quat):
#     return np.quaternion(quat.w, quat.x, -quat.y, -quat.z)

# def nwu_to_neu_vector(vector):
#     return np.array([vector[0], -vector[1], vector[2]])

def calculate_H_matrix(image_data_changing, image_data_constant):

    R_changing = quaternion.as_rotation_matrix(image_data_changing[5])
    R_constant = quaternion.as_rotation_matrix(image_data_constant[5])
    t_changing = np.array([image_data_changing[2], image_data_changing[3], image_data_changing[4]])
    t_constant = np.array([image_data_constant[2], image_data_constant[3], image_data_constant[4]])
    n = np.array([0,0,-1])
    n1 = R_changing @ n # not too sure if constant or changing
    return homography_camera_displacement(R_changing, R_constant, t_changing, t_constant, n1)

def homography_camera_displacement(R1, R2, t1, t2, n1):
    """
    Calculate homography matrix for camera displacement c1 to c2.
    
    Args:
      R1: Rotation matrix for camera1
      R2: Rotation matrix for camera2
      t1: Translation vector for camera1
      t2: Translation vector for camera2
      n1: normal vector for projection plane on camera1 coordinate
    
    Return:
      H12: homography matrix for camera displacement from 1 to 2.
      d1: distance from the plane to camera1 on camera1 coordincate.
    """
    R12 = R2 @ R1.T
    t12 = R2 @ (- R1.T @ t1) + t2
    d1  = np.inner(n1.ravel(), t1.ravel())
    H12 = R12 #+ ((t12 @ n1.T) / d1)
    H12 /= H12[2,2]
    return H12

def calculate_transformation_matrix(changing, constant):
    Hab = calculate_H_matrix(changing,constant)
    # for our data set the zs are equal
    # out = np.matmul(np.matmul(K, Hab),np.linalg.inv(K))
    out = Hab
    out /= out[2,2]
    return out

def get_img_bound(image_data):
    coords = image_data[6]
    corner_x_values = [coords[0][0][0], coords[0][-1][0], coords[-1][0][0], coords[-1][-1][0]]
    corner_y_values = [coords[0][0][1], coords[0][-1][1], coords[-1][0][1], coords[-1][-1][1]]

    # find and print min and max of all pixels
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
    return (int((max_y - min_y) * width/2) + 1,int((max_x - min_x) * width/2) + 1,3)

def transform_image(changing, constant):
    height, width, _ = changing[1].shape
    mat = calculate_transformation_matrix(changing, constant)
    changing[-1] = apply_matrix_to_vectors(changing[-1],mat)

    changing_bounds = get_img_bound(changing)
    constant_bounds = get_img_bound(constant)
    union_bounds = get_bounds_union(np.array([changing_bounds, constant_bounds]))
    new_dim = get_image_dimesion(union_bounds, width, height)
    new_img = np.zeros(new_dim, dtype=np.uint8)

    ndc_to_vp = get_ndc_to_vp_matrix(union_bounds, width, height)

    # src_corners_transformed = np.array([changing[-1][0][0], changing[-1][0][-1], changing[-1][-1][0], changing[-1][-1][-1]])
    # for corner in src_corners_transformed:
    #     vp = np.matmul(ndc_to_vp, corner)
    #     print(vp)
    imgs = [(constant[1], constant[-1]),(changing[1], changing[-1])]
    for img in imgs:
        for i in range(img[1].shape[0]):
            for j in range(img[1].shape[1]):
                vp = np.matmul(ndc_to_vp, img[1][i][j])
                if vp[0] < 0 or vp[0] >= new_dim[1] or vp[1] < 0 or vp[1] >= new_dim[0]:
                    # print(f"out of bounds: {img[1][i][j],vp}")
                    continue
                new_img[int(vp[1])][int(vp[0])] = img[0][i][j]
    return new_img


def get_ndc_to_vp_matrix(union_bounds, width, height):
    min_x = union_bounds[0]
    min_y = union_bounds[1]
    return np.array([[width/2, 0, -min_x * width/2],
                     [0, width/2, min_y * height/2],
                     [0, 0, 1]])



q_auv_cam = np.quaternion(0,0.707,-0.707,0)

def fix_quat(q_nwu_auv):
    return q_nwu_auv * q_auv_cam

data_dir = "data2"  # Path to the data directory
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
    quat = fix_quat(quat)


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
