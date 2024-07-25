import cv2
import matplotlib.pyplot as plt

# Read an image
src_img = cv2.imread("kid.jpg", cv2.IMREAD_COLOR)

# Display the original image
cv2.imshow('Original',src_img)
cv2.waitKey(0)

# Draw a rectangle (thickness is a positive integer)
imageRectangle = src_img.copy()

rect = cv2.rectangle(imageRectangle, (500, 100), (700,600), (255, 0, 255), thickness=5, lineType=cv2.LINE_8)

# Display the image
cv2.imshow('Rectangle',rect)
cv2.waitKey(0)

cv2.destroyAllWindows()

