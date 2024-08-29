import cv2
import numpy as np


def color_based_segmentation_tracking():
    cap = cv2.VideoCapture(0)

    # Define color range for tracking (e.g., red color)
    # lower_color = np.array([0, 100, 100])
    # upper_color = np.array([10, 255, 255])
    # Define color range for blue color
    # HSV ranges for blue color (lower and upper bounds)
    # lower_color = np.array([100, 150, 0])  # Lower bound for blue color
    # upper_color = np.array([140, 255, 255])  # Upper bound for blue color
    lower_color = np.array([20, 100, 100])  # Lower bound for yellow color
    upper_color = np.array([30, 255, 255])  # Upper bound for yellow color

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask
        mask = cv2.inRange(hsv, lower_color, upper_color)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show results
        cv2.imshow('Original', frame)
        cv2.imshow('Segmented', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run color-based segmentation and tracking
color_based_segmentation_tracking()
