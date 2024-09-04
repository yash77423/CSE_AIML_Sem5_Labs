# Image negative

import cv2
import matplotlib.pyplot as plt

# Read an image
img_bgr = cv2.imread('scenery.jpg', 1)
cv2.imshow("Original", img_bgr)
cv2.waitKey(0)

# get height and width of the image
height, width, _ = img_bgr.shape

for i in range(0, height - 1):
    for j in range(0, width - 1):
        # Get the pixel value
        pixel = img_bgr[i, j]

        # Negate each channel by subtracting it from 255

        # 1st index contains blue pixel
        pixel[0] = 255 - pixel[0]

        # 2nd index contains green pixel
        pixel[1] = 255 - pixel[1]

        # 3rd index contains red pixel
        pixel[2] = 255 - pixel[2]

        # Store new values in the pixel
        img_bgr[i, j] = pixel

# Display the negative transformed image
cv2.imshow("Negative", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
