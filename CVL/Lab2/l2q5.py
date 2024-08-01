# Gray level slicing

import cv2
import numpy as np

img = cv2.imread('cameraman.png', 0)
cv2.imshow("Original", img)
cv2.waitKey(0)

# To ascertain total numbers of rows and columns of the image, use size of the image
m, n = img.shape

# the lower threshold value
T1 = 100

# the upper threshold value
T2 = 180

# create an array of zeros
img_thresh_back = np.zeros((m, n), dtype=np.uint8)

for i in range(m):

    for j in range(n):

        if T1 < img[i, j] < T2:
            img_thresh_back[i, j] = 255
        else:
            img_thresh_back[i, j] = img[i, j]

# Display the image with Grey Level Slicing with Background
cv2.imshow("Gray level slicing", img_thresh_back)
cv2.waitKey(0)

