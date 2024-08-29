import cv2
import numpy as np


def hough_transform(image_path):
    def update(val):
        # Get the threshold value from the trackbar
        threshold = cv2.getTrackbarPos('Threshold', 'Hough Transform')

        # Perform Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Apply Hough Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)

        # Create a black image to draw lines on
        lines_img = np.zeros_like(gray)

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                # Draw the lines in white on the black background
                cv2.line(lines_img, (x1, y1), (x2, y2), (255), 1)

        # Convert grayscale image to BGR for visualization
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Overlay the lines on the original image
        overlay_img = np.copy(color_img)
        overlay_img[lines_img == 255] = [0, 0, 255]  # Red color for lines

        # Display the images
        cv2.imshow('Hough Transform', overlay_img)
        cv2.imshow('Lines Only', lines_img)

    # Load the image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create windows
    cv2.namedWindow('Hough Transform')
    cv2.namedWindow('Lines Only')

    # Create a trackbar for adjusting the threshold
    cv2.createTrackbar('Threshold', 'Hough Transform', 100, 200, update)

    # Initial call to update function
    update(100)

    # Wait until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run the Hough Transform
hough_transform('hough_img.jpg')
