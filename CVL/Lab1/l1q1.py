#import the cv2 library
import cv2

# The function cv2.imread() is used to read an image.
img_grayscale = cv2.imread('kid.jpg', 0)

# The function cv2.imshow() is used to display an image in a window.
cv2.imshow('grayscale image', img_grayscale)

# read the coloured checkerboard image as color
img_bgr = cv2.imread("kid.jpg", cv2.IMREAD_COLOR)
print("Extracting the RGB values of each pixel in img_bgr: ", img_bgr)

# waitKey() waits for a key press to close the window and specifies indefinite loop
cv2.waitKey(0)

# cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()

# The function cv2.imwrite() is used to write an image.
cv2.imwrite('grayscale.jpg', img_grayscale)