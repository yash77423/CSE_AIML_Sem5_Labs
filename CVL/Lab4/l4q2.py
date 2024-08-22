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


def sobel_operator(img):
    """
    Apply Sobel operator to detect edges.

    :param img: Input image (grayscale)
    :return: Edge magnitude image
    """
    # Define Sobel kernels
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

    # Compute the gradient magnitude
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize to range [0, 255]
    grad_magnitude = np.clip(grad_magnitude / grad_magnitude.max() * 255, 0, 255).astype(np.uint8)

    return grad_magnitude


# Load the image
image_path = 'valve.png'  # Update this to your image path
original_img = cv2.imread(image_path)
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# Apply Sobel operator
sobel_img = sobel_operator(gray_img)

# Convert images for display
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
sobel_img_rgb = cv2.cvtColor(sobel_img, cv2.COLOR_GRAY2RGB)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Sobel Edge Detection')
plt.imshow(sobel_img_rgb)
plt.axis('off')

plt.show()
