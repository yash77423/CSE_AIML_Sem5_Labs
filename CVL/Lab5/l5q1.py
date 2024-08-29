import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(image, ax, title):
    """Plot histogram in a given matplotlib axis."""
    ax.hist(image.ravel(), bins=256, range=[0, 256], color='black')
    ax.set_title(title)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')


def global_thresholding(image_path, initial_threshold=127, epsilon=0.1):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Initial threshold
    T = initial_threshold

    while True:
        # Segment the image
        G1 = image[image > T]
        G2 = image[image <= T]

        # Compute average intensities
        m1 = np.mean(G1) if len(G1) > 0 else 0
        m2 = np.mean(G2) if len(G2) > 0 else 0

        # Compute new threshold
        new_T = (m1 + m2) / 2

        # Check for convergence
        if abs(new_T - T) < epsilon:
            break

        T = new_T

    # Apply final threshold
    _, binary_image = cv2.threshold(image, T, 255, cv2.THRESH_BINARY)

    # Create matplotlib figures and axes
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Plot histogram of original image
    plot_histogram(image, axes[0, 0], 'Histogram of Original Image')

    # Plot histogram of binary image
    plot_histogram(binary_image, axes[0, 1], 'Histogram of Binary Image')

    # Display images
    axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB))
    axes[1, 1].set_title('Binary Image')
    axes[1, 1].axis('off')

    # Adjust layout and show
    plt.tight_layout()
    plt.show()


# Run the global thresholding
global_thresholding('gth_img.png')
