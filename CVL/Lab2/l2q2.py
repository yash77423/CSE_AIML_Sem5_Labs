# Log Transform

import cv2
import numpy as np

# Open the image
img = cv2.imread('sample.jpg')
cv2.imshow("Original", img)
cv2.waitKey(0)

# Convert image to float32 for log transformation
img_float = img.astype(np.float32)

# Apply log transform
c = 255/(np.log(1 + np.max(img_float)))
log_transformed = c * np.log(1 + img_float)

# Clip the values to be in the valid range [0, 255] and convert back to uint8
log_transformed = np.clip(log_transformed, 0, 255).astype(np.uint8)

# Show the output
cv2.imshow("Log transformed", log_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
