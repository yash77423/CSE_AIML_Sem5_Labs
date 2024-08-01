# Bright spot detection

import cv2
import numpy as np

# Load the image
image_path = 'sample.jpg'
image = cv2.imread(image_path)

# Check if image is loaded properly
if image is None:
    raise Exception("Image not found or path is incorrect")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve thresholding
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply a binary threshold to get bright spots
# You may need to adjust the threshold value based on your image
_, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

# Find contours of the bright spots
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
result_image = image.copy()
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2

# Show the images
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', thresholded)
cv2.imshow('Detected Bright Spots', result_image)

# Wait until a key is pressed and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
