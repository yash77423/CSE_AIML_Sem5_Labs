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


def gaussian_blur(img, kernel_size, sigma):
    """
    Apply Gaussian blur to an image.

    :param img: Input image (grayscale)
    :param kernel_size: Size of the Gaussian kernel
    :param sigma: Standard deviation of the Gaussian kernel
    :return: Blurred image
    """
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size),
        dtype=np.float32
    )
    kernel /= np.sum(kernel)  # Normalize the kernel

    return apply_kernel(img, kernel)


def unsharp_masking(img, kernel_size=5, sigma=1.0, alpha=1.5):
    """
    Apply unsharp masking to an image.

    :param img: Input image (grayscale)
    :param kernel_size: Size of the Gaussian kernel
    :param sigma: Standard deviation of the Gaussian kernel
    :param alpha: Amount of sharpening (scaling factor for the mask)
    :return: Image with enhanced sharpness
    """
    # Apply Gaussian blur to create the blurred version of the image
    blurred_img = gaussian_blur(img, kernel_size, sigma)

    # Compute the mask by subtracting the blurred image from the original
    mask = img - blurred_img

    # Enhance the original image by adding the scaled mask
    sharpened_img = np.clip(img + alpha * mask, 0, 255).astype(np.uint8)

    return sharpened_img


# Load the image
image_path = 'usm-img.jpg'  # Update this to your image path
original_img = cv2.imread(image_path)
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# Apply unsharp masking
sharpened_img = unsharp_masking(gray_img, kernel_size=5, sigma=1.0, alpha=1.5)

# Convert images for display
original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
sharpened_img_rgb = cv2.cvtColor(sharpened_img, cv2.COLOR_GRAY2RGB)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Unsharp Masked Image')
plt.imshow(sharpened_img_rgb)
plt.axis('off')

plt.show()
