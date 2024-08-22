import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_kernel(img, kernel):
    """
    Apply convolution with a given kernel to an image.

    :param img: Input image (grayscale)
    :param kernel: Convolution kernel
    :return: Image after applying the kernel
    """
    # Image dimensions
    height, width = img.shape
    k_height, k_width = kernel.shape
    pad_height, pad_width = k_height // 2, k_width // 2

    # Pad the image to handle borders
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Output image
    output_img = np.zeros_like(img, dtype=np.float32)

    # Apply convolution
    for y in range(height):
        for x in range(width):
            region = padded_img[y:y + k_height, x:x + k_width]
            output_img[y, x] = np.sum(region * kernel)

    return output_img


def laplacian_filter(img):
    """
    Apply Laplacian filter to an image and enhance edges.

    :param img: Input image (grayscale)
    :return: Image with enhanced edges
    """
    # Define Laplacian kernel
    laplacian_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)

    # Apply Laplacian kernel
    laplacian_img = apply_kernel(img, laplacian_kernel)

    # Compute absolute value to enhance edges
    laplacian_img = np.abs(laplacian_img)

    # Normalize the result to the range [0, 255]
    laplacian_img = np.clip(laplacian_img, 0, 255)  # Ensure values are in range
    laplacian_img = (laplacian_img / laplacian_img.max() * 255).astype(np.uint8)

    # Invert image for better visualization of edges
    laplacian_img = laplacian_img

    return laplacian_img


# Load the image
image_path = 'astroflag.jpg'  # Update this to your image path
original_img = cv2.imread(image_path)
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# Apply Laplacian filter
laplacian_img = laplacian_filter(gray_img)

# Convert images for display
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
laplacian_img_rgb = cv2.cvtColor(laplacian_img, cv2.COLOR_GRAY2RGB)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Laplacian Filtered Image')
plt.imshow(laplacian_img_rgb)
plt.axis('off')

plt.show()
