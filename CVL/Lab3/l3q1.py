import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_clahe(image):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image."""
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Split LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L channel back with A and B channels
    limg = cv2.merge((cl, a, b))

    # Convert LAB back to BGR
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)


def plot_histogram(image, title, subplot_position):
    """Plots the histogram of an image."""
    plt.subplot(subplot_position)
    plt.hist(image.ravel(), bins=256, range=(0, 256))
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')


def plot_images_and_histograms(original_image, clahe_image):
    """Plots the original and CLAHE images with their histograms."""
    # Convert images to grayscale for histogram plotting
    gray_original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(12, 10))

    # Plot original image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Plot CLAHE image
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2RGB))
    plt.title('CLAHE Image')
    plt.axis('off')

    # Plot original image histogram
    plot_histogram(gray_original_image, 'Histogram of Original Image', 223)

    # Plot CLAHE image histogram
    plot_histogram(gray_clahe_image, 'Histogram of CLAHE Image', 224)

    plt.tight_layout()
    plt.show()


# Load the image
image_path = 'sample1.jpg'
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image not found or unable to open.")

# Apply CLAHE
clahe_image = apply_clahe(image)

# Plot images and histograms
plot_images_and_histograms(image, clahe_image)
