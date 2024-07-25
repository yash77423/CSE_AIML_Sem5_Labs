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
plt.figure(figsize=(18, 5))
plt.subplot(141); plt.imshow(rot_img_1); plt.title("Rotate 90° CW")

plt.subplot(142); plt.imshow(rot_img_2); plt.title("Rotate 180°")

plt.subplot(143); plt.imshow(rot_img_3); plt.title("Rotate 90° ACW")

plt.subplot(144); plt.imshow(src_img); plt.title("Original")