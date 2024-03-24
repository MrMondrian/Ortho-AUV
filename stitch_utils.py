import numpy as np  # Assuming you want to save image data as NumPy arrays
import quaternion


def pixel_coords(image_shape):
    """
    Returns numpy array of pixel indicies in homogenous coordinates
    """
    coords = np.indices(image_shape[:2])
    out = np.zeros((image_shape[0],image_shape[1],3))
    out[:,:,0] = coords[1]
    out[:,:,1] = coords[0]
    out[:,:,2] = 1
    return out


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

def calculate_H_matrix(changing, constant):

    """
    Calculates the homography matrix H that maps points from the changing image to the constant image.
    """

    pos_changing = quaternion.rotate_vectors(changing["quat"],changing["position"])
    pos_constant = quaternion.rotate_vectors(constant["quat"],constant["position"])
    n = np.array([0,0,1])
    n = quaternion.rotate_vectors(changing["quat"],n).reshape((3,1)) # not too sure if constant or changing

    q_b_a = constant["quat"].inverse() * changing["quat"]
    pos_a_b =  pos_constant - quaternion.rotate_vectors(q_b_a, pos_changing)
    d  = -30
    R_b_a = quaternion.as_rotation_matrix(q_b_a)
    H = R_b_a - ((pos_a_b.reshape((3,1)) @ n.T) / d)
    H /= H[2,2]
    return H

def calculate_transformation_matrix(changing, constant):
    """
    Calculates the transformation matrix that maps points from the changing image to the constant image.

    return K * H * K^-1
    where H is calculated using calculate_H_matrix
    """


    Hab = calculate_H_matrix(changing,constant)
    # for our data set the zs are equal
    out = np.matmul(np.matmul(changing["intrinsics"], Hab),np.linalg.inv(constant["intrinsics"]))
    out /= out[2,2]
    return out

def get_img_bound(coords):
    """
    Returns the min and max x and y values of the image in pixel coordinates
    """

    corner_x_values = [coords[0][0][0], coords[0][-1][0], coords[-1][0][0], coords[-1][-1][0]]
    corner_y_values = [coords[0][0][1], coords[0][-1][1], coords[-1][0][1], coords[-1][-1][1]]

    # find and print min and max of all pixels
    return np.array([min(corner_x_values), max(corner_x_values), min(corner_y_values), max(corner_y_values)])

def get_bounds_union(bounds):
    """
    Returns the union of the bounds of multiple images
    """

    min_x = np.min(bounds[:,0])
    max_x = np.max(bounds[:,1])
    min_y = np.min(bounds[:,2])
    max_y = np.max(bounds[:,3])
    return np.array([min_x, max_x, min_y, max_y])

def get_image_dimesion(union_bounds):
    """
    Returns the dimensions of the stitched image
    """
    min_x = union_bounds[0]
    max_x = union_bounds[1]
    min_y = union_bounds[2]
    max_y = union_bounds[3]
    return int((max_y - min_y)) + 1, int((max_x - min_x)) + 1, 3

def offset_pixels(coords, union_bounds):
    """
    Offsets the pixel coordinates of an image to align with the union bounds
    """
    min_x = union_bounds[0]
    min_y = union_bounds[2]
    coords[:,:,0] -= min_x
    coords[:,:,1] -= min_y

def transform_image(changing, constant):
    """
    Transforms the pixel coordinates of the changing image to align with the constant image
    """
    mat = calculate_transformation_matrix(changing, constant)
    changing["pixel_coords"] = apply_matrix_to_vectors(changing["pixel_coords"],mat)

def render_image(constant,changing_imgs):
    """
    Renders the images into a single stitched image
    """

    # get the union of the bounds of all images 
    bounds = [get_img_bound(constant["pixel_coords"])]
    for img in changing_imgs:
        bounds.append(get_img_bound(img["pixel_coords"]))
    union_bounds = get_bounds_union(np.array(bounds))

    # offset the pixel coordinates of all images to align with the union bounds
    for img in changing_imgs:
        offset_pixels(img["pixel_coords"], union_bounds)
    offset_pixels(constant["pixel_coords"], union_bounds)

    # create a new image with the dimensions of the union bounds
    new_dim = get_image_dimesion(union_bounds)
    new_img = np.zeros(new_dim, dtype=np.uint8)

    # render the images into the new image
    imgs = [constant] + changing_imgs
    for img in imgs:
        img["pixel_coords"] = img["pixel_coords"][:,:,:2]
        img["pixel_coords"] = img["pixel_coords"].astype(int)
        new_img[img["pixel_coords"][:,:,1],img["pixel_coords"][:,:,0],:] = img["image_data"]
    return new_img