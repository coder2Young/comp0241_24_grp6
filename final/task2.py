import cv2
import numpy as np
import os
import time

debug = False

def get_mask(image):
    # Convert image to HSV color space
    # For acceleration, we can use cv2.COLOR_BGR2HSV directly
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Method1: Color Thresholding
    lower_hsv = np.array([80, 10, 10])
    upper_hsv = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Noise removal
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Method2: Enclosing Circle
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 5000
    max_area = 0
    max_circle = None
    center = None

    for contour in contours:
        area = cv2.contourArea(contour)
        box = cv2.minAreaRect(contour)
        if box[1][0] == 0 or box[1][1] == 0:
            continue
        roi = max(box[1][0] / box[1][1], box[1][1] / box[1][0])
        if roi > 1.5 or area < min_area:
            continue
        # Find minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if area > max_area:
            max_area = area
            max_circle = (x, y, radius)

    if max_circle:
        (x, y, radius) = max_circle
        center = (int(x), int(y))
        radius = int(radius - 10)

        # Create mask
        height, width = image.shape[:2]
        maskImg = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(maskImg, center, radius, 255, thickness=-1)
    else:
        # Return empty mask
        height, width = image.shape[:2]
        maskImg = np.zeros((height, width), dtype=np.uint8)

    return image, maskImg, center, radius

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Get the width and height of frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Define the codec and create VideoWriter object
dataset_path = "./Dataset/grp6"
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(os.path.join(dataset_path,"AO_cap.mp4"), fourcc, 20.0, (frame_width, frame_height))

start_time = time.time()
centroids = []
WINDOW_SIZE = 10

cv2.waitKey(500)
while True:
    ret, frame = cap.read()

    # Up-down flip
    frame = cv2.flip(frame, 0)

    if not ret:
        print("Error: Could not read frame.")
        break

    # Get mask
    frame, mask, centroid, radius = get_mask(frame)

    if centroid is not None:
        if len(centroids) >= WINDOW_SIZE:
            centroids.pop(0)
            
        centroids.append(centroid)
        # Calculate the average of the last 10 centroids
        avg_centroid = np.mean(centroids, axis=0)
        cv2.circle(frame, (int(avg_centroid[0]), int(avg_centroid[1])), 5, (0, 0, 255), -1)
        # Put text about the centroid
        cv2.putText(frame, f"Centroid: {avg_centroid}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate time elapsed
    time_elapsed = time.time() - start_time
    # Mark time_elpased on video
    cv2.putText(frame, f"Time Elapsed: {time_elapsed:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the frame into the file 'output.avi'
    out.write(frame)
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Calculate SSD aroung centroid
    if len(centroids) == WINDOW_SIZE:
        ssd = np.sum(np.square(centroids - avg_centroid), axis=1)
        if debug:
            print(f"SSD: {ssd}")

    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()