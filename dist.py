import cv2
import numpy as np

# Termination criteria for subpixel corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)



# Load the chessboard image
img = cv2.imread('good.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
# binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
# binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# binary = cv2.GaussianBlur(binary, (5, 5), 0)

color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
cv2.imwrite('binary.jpg', color)

marked = img = cv2.imread('marked.png')
def find_redish_pixels(image):
    """
    Finds all pixel coordinates that are red-ish in the given image.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        list: A list of (x, y) coordinates representing the red-ish pixels.
    """
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red-ish color range in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Create a mask for the red-ish color range
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Find the coordinates of non-zero pixels (red-ish pixels) in the mask
    redish_pixels = cv2.findNonZero(mask)
    # Convert the coordinates to a list of tuples
    redish_pixel_coords = [[arr[0][0], arr[0][1]] for arr in redish_pixels]

    return np.array(redish_pixel_coords, dtype=np.float32)
corners = find_redish_pixels(marked)
refined_corners = cv2.cornerSubPix(binary, corners, (11, 11), (-1, -1), criteria)

def sort_pixels(pixels):
    pixels = pixels[pixels[:, 0].argsort()]#[pixels[:, 0].argsort()]
    rows = 6
    for i in range(0, len(pixels), rows):
        pixels[i:i+rows] = pixels[i:i+rows][pixels[i:i+rows, 1].argsort()[::-1]]
    return pixels

refined_corners = sort_pixels(refined_corners)

ret = True
# Draw and display the corners

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Append the object points and image points for the current image
objpoints.append(objp)
imgpoints.append(refined_corners)

img = cv2.drawChessboardCorners(img, (6,9), refined_corners, ret)
cv2.imwrite('checkered.jpg', img)

guess = np.array([[358.0158, 0.00000000e+00, 640],
                [0.00000000e+00, 358.0158, 360],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
ret, mtx_good, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
ret, mtx, dist_good, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], guess, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_FOCAL_LENGTH)

print("Camera Distortion Coefficients: ", dist_good[0])
print("Camera Matrix: ", mtx)
undistorted_img = cv2.undistort(img, mtx_good, dist_good)

# Save the undistorted image
cv2.imwrite('undistorted.jpg', undistorted_img)
