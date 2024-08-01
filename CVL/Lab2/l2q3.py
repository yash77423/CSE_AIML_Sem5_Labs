# Power-law (Gamma) Transform

import cv2
import numpy as np

# Open the image
img = cv2.imread('sample.jpg')
cv2.imshow("Original", img)
cv2.waitKey(0)

# Trying 4 gamma values
for gamma in [0.1, 0.5, 1.2, 2.2]:
    # Apply gamma correction
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')

    # Show edited images
    cv2.imshow("Gamma transformed", gamma_corrected)
    cv2.waitKey(0)

cv2.destroyAllWindows()
