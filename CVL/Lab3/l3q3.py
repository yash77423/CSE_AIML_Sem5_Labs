import cv2
import numpy as np

def apply_filter(image, kernel):
    """Applies a filter to an image using convolution."""
    return cv2.filter2D(image, -1, kernel)

def create_average_kernel(kernel_size):
    """Creates an average (smoothing) kernel."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    return kernel

def create_sharpening_kernel():
    """Creates a sharpening kernel."""
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    return kernel

def create_gaussian_kernel(kernel_size, sigma):
    """Generates a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) *
                    np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma**2)),
        (kernel_size, kernel_size)
    )
    return kernel / np.sum(kernel)  # Normalize the kernel

def median_blur(image, kernel_size):
    """Applies median blur to the image."""
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    output = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Extract the neighborhood
            neighborhood = padded_image[y:y+kernel_size, x:x+kernel_size]
            # Compute the median of the neighborhood
            output[y, x] = np.median(neighborhood)

    return output

def min_blur(image, kernel_size):
    """Applies min blur to the image."""
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=255)
    output = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            neighborhood = padded_image[y:y+kernel_size, x:x+kernel_size]
            output[y, x] = np.min(neighborhood)

    return output

def max_blur(image, kernel_size):
    """Applies max blur to the image."""
    pad = kernel_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    output = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            neighborhood = padded_image[y:y+kernel_size, x:x+kernel_size]
            output[y, x] = np.max(neighborhood)

    return output

# Load the image
image_path = 'sample.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not found or unable to open.")

# Define kernel sizes and parameters
kernel_size = 5
sigma = 1.0

# Create kernels
average_kernel = create_average_kernel(kernel_size)
sharpening_kernel = create_sharpening_kernel()
gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)

# Apply filters
smoothed_image = apply_filter(image, average_kernel)
sharpened_image = apply_filter(image, sharpening_kernel)
median_blurred_image = median_blur(image, kernel_size)
min_blurred_image = min_blur(image, kernel_size)
max_blurred_image = max_blur(image, kernel_size)
gaussian_blurred_image = apply_filter(image, gaussian_kernel)

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Smoothed Image', smoothed_image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.imshow('Median Blurred Image', median_blurred_image)
cv2.imshow('Min Blurred Image', min_blurred_image)
cv2.imshow('Max Blurred Image', max_blurred_image)
cv2.imshow('Gaussian Blurred Image', gaussian_blurred_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
