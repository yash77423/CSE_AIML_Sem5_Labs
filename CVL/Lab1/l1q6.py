# Python program to explain cv2.rotate() method

# importing cv2 and matplotlib
import cv2
import matplotlib.pyplot as plt

# Reading an image in default mode
src_img = cv2.imread("kid.jpg")


rot_img_1 = cv2.rotate(src_img, cv2.ROTATE_90_CLOCKWISE)  # Using cv2.ROTATE_90_CLOCKWISE rotate by 90 degrees clockwise
rot_img_2 = cv2.rotate(src_img, cv2.ROTATE_180)  # Using cv2.ROTATE_180 rotate by 180 degrees clockwise
rot_img_3 = cv2.rotate(src_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Using cv2.ROTATE_90_COUNTERCLOCKWISE rotate by 270 degrees clockwise

# Show the images

cv2.imshow('Rotate 90° CW', rot_img_1)
cv2.waitKey(0)
cv2.imshow('Rotate 180°', rot_img_2)
cv2.waitKey(0)
cv2.imshow('Rotate 90° ACW', rot_img_3)
cv2.waitKey(0)
cv2.imshow('Original', src_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
