import cv2
import matplotlib.pyplot as plt

# Read an image
src_img = cv2.imread("kid.jpg", cv2.IMREAD_COLOR)

# Display the original image
plt.imshow(src_img[:,:,::-1])

# Draw a rectangle (thickness is a positive integer)
imageRectangle = src_img.copy()

cv2.rectangle(imageRectangle, (500, 100), (700,600), (255, 0, 255), thickness=5, lineType=cv2.LINE_8)

# Display the image
plt.imshow(imageRectangle[:,:,::-1])

