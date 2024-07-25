# Resizing image

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading the tomato_image
kid_image = cv2.imread("kid.jpg", 1)
print(kid_image.shape)

half = cv2.resize(kid_image, (0, 0), fx = 0.5, fy = 0.5)
bigger = cv2.resize(kid_image, (1050, 1610))
# while using the cv2.resize() function,
# the tuple passed for determining the size of the new image ((1050, 1610) in this case)
# follows the order (width, height) unlike as expected (height, width)

stretch_near = cv2.resize(kid_image, (780, 540), interpolation = cv2.INTER_LINEAR)


Titles =["Original", "Half", "Bigger", "Interpolation Nearest"]
kid_images =[kid_image, half, bigger, stretch_near]
count = 4

for i in range(count):
	plt.subplot(2, 2, i + 1)
	plt.title(Titles[i])
	plt.imshow(kid_images[i])

plt.show()
