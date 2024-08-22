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


def compute_gradients(img):
    """
    Compute gradients of an image using Sobel operators.

    :param img: Input image (grayscale)
    :return: Gradients in x and y directions, and gradient magnitude
    """
    # Define Sobel kernels for x and y gradients
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    # Apply Sobel kernels
    grad_x = apply_kernel(img, sobel_x)
    grad_y = apply_kernel(img, sobel_y)

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize the result to the range [0, 255]
    grad_magnitude = np.clip(grad_magnitude / grad_magnitude.max() * 255, 0, 255).astype(np.uint8)

    return grad_x, grad_y, grad_magnitude


# Load the image
image_path = 'astroflag.jpg'  # Update this to your image path
original_img = cv2.imread(image_path)
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# Compute gradients
grad_x, grad_y, grad_magnitude = compute_gradients(gray_img)

# Convert images for display
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
grad_x_rgb = cv2.cvtColor(grad_x, cv2.COLOR_GRAY2RGB)
grad_y_rgb = cv2.cvtColor(grad_y, cv2.COLOR_GRAY2RGB)
grad_magnitude_rgb = cv2.cvtColor(grad_magnitude, cv2.COLOR_GRAY2RGB)

# Display the images
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(original_img_rgb)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Gradient X')
plt.imshow(grad_x_rgb)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Gradient Y')
plt.imshow(grad_y_rgb)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Gradient Magnitude')
plt.imshow(grad_magnitude_rgb)
plt.axis('off')

plt.show()
