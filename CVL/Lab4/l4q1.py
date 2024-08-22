
# importing libraries
import cv2
import numpy as np
image = cv2.imread('taj.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Bilateral Blur
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Blurring', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()
