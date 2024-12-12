import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_mask(frame):
    """
    Generate a mask for the desired region and calculate its centroid and radius.
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV range to isolate the desired color
    lower_hsv = np.array([80, 10, 10])
    upper_hsv = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Apply morphological operations to remove noise
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 5000  # Minimum area of the detected region
    max_area = 0
    max_circle = None

    for contour in contours:
        area = cv2.contourArea(contour)
        box = cv2.minAreaRect(contour)
        if box[1][0] == 0 or box[1][1] == 0:
            continue
        roi = max(box[1][0] / box[1][1], box[1][1] / box[1][0])
        if roi > 1.5 or area < min_area:
            continue
        # Find the minimum enclosing circle for the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if area > max_area:
            max_area = area
            max_circle = (x, y, radius)

    if max_circle:
        return mask, (int(max_circle[0]), int(max_circle[1])), max_circle[2]
    else:
        return mask, None, None

# Read the raw format color image
color_raw_path = "./Dataset/grp6/task2/bottom_color.raw"
color_width, color_height = 1280, 800  # Adjust based on the actual resolution
with open(color_raw_path, "rb") as f:
    color_raw = np.frombuffer(f.read(), dtype=np.uint8)

# Read the raw format depth image
depth_raw_path = "./Dataset/grp6/task2/bottom_Depth.raw"
depth_width, depth_height = 848, 480  # Adjust based on the actual resolution
with open(depth_raw_path, "rb") as f:
    depth_raw = np.frombuffer(f.read(), dtype=np.uint16)

# Reshape the raw data to H x W x C format (assume RGB format)
color_image = color_raw.reshape((color_height, color_width, 3))
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

# Generate a mask and find the centroid of the desired region
_, centroid, _ = get_mask(color_image)
x_c, y_c = centroid
# Draw a circle at the centroid
cv2.circle(color_image, centroid, 5, (255, 0, 0), -1)
cv2.putText(color_image, f"Centroid {centroid}", (x_c + 10, y_c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
# Save the image with the centroid marked
plt.imshow(color_image)
plt.axis("off")
plt.savefig("./Dataset/grp6/task2/task2_centroid.png", bbox_inches="tight")

# Map the color image coordinates to depth image coordinates
x_d = int(x_c * depth_width / color_width)
y_d = int(y_c * depth_height / color_height)

# Reshape the raw depth data to H x W format
depth_image = depth_raw.reshape((depth_height, depth_width))

# Retrieve the depth value at the target coordinates (convert to meters)
depth_value = depth_image[y_d, x_d] / 1000  # Depth in meters
print(f"Height of AO is {depth_value} meters")