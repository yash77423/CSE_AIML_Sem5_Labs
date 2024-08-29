import cv2
import numpy as np


def canny_edge_detection(image, low_threshold, high_threshold):
    # Gaussian filter
    blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

    # Gradient calculation
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    angle = np.arctan2(grad_y, grad_x) * (180.0 / np.pi) % 180

    # Non-maximum suppression
    suppressed = np.zeros_like(magnitude)

    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            current_angle = angle[i, j]

            if (0 <= current_angle < 22.5) or (157.5 <= current_angle <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif (22.5 <= current_angle < 67.5):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif (67.5 <= current_angle < 112.5):
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            else:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]

    # Double threshold
    strong = (suppressed >= high_threshold)
    weak = ((suppressed >= low_threshold) & (suppressed < high_threshold))

    # Edge tracking by hysteresis
    edges = np.zeros_like(suppressed)
    strong_i, strong_j = np.nonzero(strong)
    weak_i, weak_j = np.nonzero(weak)

    for i, j in zip(strong_i, strong_j):
        edges[i, j] = 255
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                if 0 <= x < suppressed.shape[0] and 0 <= y < suppressed.shape[1]:
                    if weak[x, y]:
                        edges[x, y] = 255

    return edges


def update_thresholds(val):
    global low_threshold, high_threshold
    low_threshold = cv2.getTrackbarPos('Low Threshold', 'Edge Detection')
    high_threshold = cv2.getTrackbarPos('High Threshold', 'Edge Detection')


def process_camera():
    global low_threshold, high_threshold
    low_threshold = 50
    high_threshold = 150

    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Edge Detection')
    cv2.createTrackbar('Low Threshold', 'Edge Detection', low_threshold, 255, update_thresholds)
    cv2.createTrackbar('High Threshold', 'Edge Detection', high_threshold, 255, update_thresholds)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = canny_edge_detection(gray, low_threshold, high_threshold)

        # Convert edge-detected image to BGR for display
        edges_bgr = cv2.cvtColor(edges.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Display the original and edge-detected images in separate windows
        cv2.imshow('Original Video', frame)
        cv2.imshow('Edge Detection', edges_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the camera processing
process_camera()
