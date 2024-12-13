import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Function to update the weighted average for smoothing centroid or radius
def update_weighted_average(new_value, value_window, weight_window, max_distance):
    """
    Smooth the input values (centroid or radius) using a weighted average.
    """
    # Initialize the weighted average
    if len(value_window) == 0:
        current_average = np.array(new_value)
    else:
        weighted_sum = np.sum([w * np.array(v) for v, w in zip(value_window, weight_window)], axis=0)
        weight_sum = np.sum(weight_window)
        current_average = weighted_sum / weight_sum

    # Compute the distance of the new value from the current average
    distance = np.linalg.norm(np.array(new_value) - current_average)
    new_weight = 1 / (1 + distance / max_distance)

    # Update the sliding window
    value_window.append(new_value)
    weight_window.append(new_weight)

    # Recalculate the smoothed value
    weighted_sum = np.sum([w * np.array(v) for v, w in zip(value_window, weight_window)], axis=0)
    weight_sum = np.sum(weight_window)
    smoothed_value = weighted_sum / weight_sum

    return smoothed_value

# Function to generate a mask for the Earth-like region and calculate its centroid and radius
def get_mask(frame):
    """
    Extract the mask of the Earth-like region and return its centroid and radius.
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Thresholding to isolate the desired color range
    lower_hsv = np.array([80, 10, 10])
    upper_hsv = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Morphological operations to remove noise
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 5000
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
        # Find the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if area > max_area:
            max_area = area
            max_circle = (x, y, radius)

    if max_circle:
        return mask, (int(max_circle[0]), int(max_circle[1])), max_circle[2]
    else:
        return mask, None, None

video_name = "task3e_close"

# Enable debug mode
debug = False

# Video file path and capture initialization
video_path = "./Dataset/grp6/task3/" + video_name + ".mp4"
cap = cv2.VideoCapture(video_path)
FPS = cap.get(cv2.CAP_PROP_FPS)
if debug:
    print(f"FPS: {FPS}")

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Parameters
WINDOW_SIZE = 50
MAX_WEIGHT_DISTANCE = 500
MAX_TIME = 700  # 12 minutes in seconds, limite process time, between one rotation and two
start_time = 10  # Start analyzing at 10 seconds, to avoid the initial noise

# Initialize smoothing windows for centroid and radius
centroid_window = deque(maxlen=WINDOW_SIZE)
weight_window = deque(maxlen=WINDOW_SIZE)
radius_window = deque(maxlen=WINDOW_SIZE)

rotation_periods = []  # Store rotation periods
SSD = []  # Store SSD values
last_period_time = None

# Calculate frame numbers
init_frame_num = int(FPS * start_time)
drop_frame_num = int(FPS * (start_time - 5))

# Initial smoothing for centroid and radius
for i in range(init_frame_num):
    ret, frame = cap.read()
    _, centroid, radius = get_mask(frame)
    if centroid and radius:
        smoothed_centroid = update_weighted_average(centroid, centroid_window, weight_window, MAX_WEIGHT_DISTANCE)
        smoothed_radius = update_weighted_average(radius, radius_window, weight_window, MAX_WEIGHT_DISTANCE)
        if i > drop_frame_num:
            init_centroid = smoothed_centroid
            init_radius = smoothed_radius
            # Save the initial frame if the centroid and radius are close to initial values
            if np.linalg.norm(init_centroid - centroid) < 50 and abs(init_radius - radius) < 50:
                init_frame = frame

if debug:
    print(f"Initial centroid: {init_centroid}, radius: {init_radius}")

# Process video frames
current_frame = 0
while True:
    process_start_time = cv2.getTickCount()
    ret, frame = cap.read()
    if not ret:
        break

    current_frame += 1

    # Get mask, centroid, and radius for the current frame
    mask, centroid, radius = get_mask(frame)
    if centroid is None or radius is None:
        SSD.append(SSD[-1] if len(SSD) > 0 else 0)
        continue

    # Smooth centroid and radius
    smoothed_centroid = update_weighted_average(centroid, centroid_window, weight_window, MAX_WEIGHT_DISTANCE)

    # Align the ROI region
    x, y = map(int, smoothed_centroid)
    init_x, init_y = map(int, init_centroid)

    init_roi = cv2.getRectSubPix(init_frame, (2 * int(init_radius), 2 * int(init_radius)), (init_x, init_y))
    current_roi = cv2.getRectSubPix(frame, (2 * int(init_radius), 2 * int(init_radius)), (x, y))
    init_mask = np.zeros_like(init_roi, dtype=np.uint8)
    cv2.circle(init_mask, (int(init_radius), int(init_radius)), int(init_radius), (255, 255, 255), -1)

    # Convert mask to 1 channel
    init_mask = cv2.cvtColor(init_mask, cv2.COLOR_BGR2GRAY)

    # Apply mask to set a circle
    init_roi = cv2.bitwise_and(init_roi, init_roi, mask=init_mask)
    current_roi = cv2.bitwise_and(current_roi, current_roi, mask=init_mask)

    # Visualize the ROI if in debug mode
    if debug:
        cv2.imshow("Initial ROI", init_roi)
        cv2.imshow("Current ROI", current_roi)

    # Calculate SSD
    init_roi = cv2.GaussianBlur(init_roi, (21, 21), 0)
    current_roi = cv2.GaussianBlur(current_roi, (21, 21), 0)
    ssd = np.sum((init_roi - current_roi) ** 2)
    SSD.append(ssd)

    # Break conditions
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if current_frame > MAX_TIME * FPS:
        break
    process_end_time = cv2.getTickCount()
    process_time = (process_end_time - process_start_time) / cv2.getTickFrequency()
    frame_time = 1 / FPS
    if current_frame % (FPS * 3) == 0:
        print(f"Frame {current_frame}: Process time: {process_time:.4f}s, Frame time: {frame_time:.4f}s")

cap.release()
cv2.destroyAllWindows()

# Calculate rotation period based on SSD
SS = SSD[drop_frame_num:] # Drop the first few frames
min_index = np.argmin(SS) + drop_frame_num
min_value = SSD[min_index]
period = min_index / FPS
print(f"Rotation period: {period:.2f}s")

# Plot SSD over time
SSD = np.array(SSD)
plt.plot(SSD)
plt.xlabel("Frame")
plt.ylabel("SSD")
plt.title(f"SSD over time for {video_name}")
plt.text(min_index-5000, min_value, f"Min SSD at {min_index}th frame\nFPS is {FPS}", fontsize=12, ha='center', color='red')
plt.text(len(SSD) // 2, min(SSD), f"Rotation period: {period:.2f}s", fontsize=12, ha='center', color='red') 
plt.savefig(f"./Dataset/grp6/task3/{video_name}_SSD.png")
plt.show()