import cv2
import numpy as np
import os
import time
from collections import deque

# Function to generate a mask and calculate the centroid and radius of the Earth-like region
def get_mask(frame):
    """
    Generate a mask for the desired region and calculate its centroid and radius.
    """
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold to isolate the desired color range
    lower_hsv = np.array([80, 10, 10])
    upper_hsv = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Noise removal with morphological operations
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours
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

# Function to smooth centroid updates using a weighted average
def update_weighted_average(new_centroid, centroid_window, weight_window, max_distance):
    """
    Smooth centroid updates using a weighted average.
    """
    # Initialize the current average
    if len(centroid_window) == 0:
        current_average = np.array(new_centroid)
    else:
        weighted_sum = np.sum([w * np.array(c) for c, w in zip(centroid_window, weight_window)], axis=0)
        weight_sum = np.sum(weight_window)
        current_average = weighted_sum / weight_sum

    # Compute distance from the new centroid to the current average
    distance = np.linalg.norm(np.array(new_centroid) - current_average)
    new_weight = 1 / (1 + distance / max_distance)

    # Update sliding windows
    centroid_window.append(new_centroid)
    weight_window.append(new_weight)

    # Recalculate the smoothed centroid
    weighted_sum = np.sum([w * np.array(c) for c, w in zip(centroid_window, weight_window)], axis=0)
    weight_sum = np.sum(weight_window)
    smoothed_centroid = weighted_sum / weight_sum

    return smoothed_centroid

# Enable debug mode
debug = True
# Write the output to a video file
write = False
# Set video source: from camera or file
from_camera = False

if from_camera:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("./Dataset/grp6/task2/task2.mp4")

FPS = cap.get(cv2.CAP_PROP_FPS)

# Check if the video source is available
if not cap.isOpened():
    print("Error: Could not open video.")
    exit(1)

# Get frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Set up the video writer
dataset_path = "./Dataset/grp6/task2"
if write:
    file_name = f"task2_AO_cap_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(os.path.join(dataset_path, file_name), fourcc, FPS, (frame_width, frame_height))

# Initialize variables
start_time = time.time()
centroids = []
WINDOW_SIZE = 50
MAX_WEIGHT_DISTANCE = 500
centroid_window = deque(maxlen=WINDOW_SIZE)
weight_window = deque(maxlen=WINDOW_SIZE)
cap_centroid = []
cap_time = []

# Process frames in the video
frame_count = 0
while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        print("Error: Could not read frame.")
        break

    if from_camera:
        frame = cv2.flip(frame, 0)

    _, centroid, radius = get_mask(frame)
    if centroid is not None:
        # Smooth the centroid position
        smoothed_centroid = update_weighted_average(centroid, centroid_window, weight_window, MAX_WEIGHT_DISTANCE)

        # Visualize the smoothed centroid
        smoothed_centroid_int = tuple(map(int, smoothed_centroid))
        if len(cap_centroid) == 0:
            cv2.circle(frame, smoothed_centroid_int, 5, (0, 255, 0), -1)  # Currnt for Green dot
            cv2.putText(frame, f"Current Centroid: {smoothed_centroid_int}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            for i in range(len(cap_centroid)):
                prev_centroid = cap_centroid[i]
                time_elapsed = cap_time[i]
                cv2.circle(frame, prev_centroid, 5, (0, 0, 255), -1) # Prev for Red dot
                cv2.putText(frame, f"Centroid{i}: {prev_centroid} at Time {time_elapsed:.2f}", (10, 30*(i+3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(frame, f"Current Centroid: {smoothed_centroid_int}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(frame, smoothed_centroid_int, 5, (0, 255, 0), -1)
        if debug:
            # Draw enclosing circle
            cv2.circle(frame, smoothed_centroid_int, int(radius), (255, 0, 0), 2)

    # Calculate elapsed time
    time_elapsed = time.time() - start_time

    # Write frame to output file if recording is enabled
    if write:
        out.write(frame)

    # Display the current frame
    cv2.imshow('frame', frame)

    # Handle user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Capture a frame
        ellpased_time = frame_count / FPS
        cap_centroid.append(smoothed_centroid_int)
        cap_time.append(ellpased_time)
    elif key == ord('s'):
        # Save the frame as an image
        cv2.imwrite(os.path.join(dataset_path, f"task2_{time.strftime('%Y%m%d-%H%M%S')}.png"), frame)
        print("Frame saved.")

# Release resources
cap.release()
cv2.destroyAllWindows()