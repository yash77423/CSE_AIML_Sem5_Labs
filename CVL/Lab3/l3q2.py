import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.ravel()
    return hist

def compute_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf[-1]  # Normalize to match range of pixel values
    return cdf, cdf_normalized

def map_histogram(source_cdf, target_cdf):
    source_cdf = source_cdf.astype('float32')
    target_cdf = target_cdf.astype('float32')
    mapping = np.interp(source_cdf, target_cdf, np.arange(256))
    return mapping

def histogram_matching(source_img, target_img):
    # Compute histograms and CDFs
    source_hist = compute_histogram(source_img)
    target_hist = compute_histogram(target_img)
    source_cdf, _ = compute_cdf(source_hist)
    target_cdf, _ = compute_cdf(target_hist)

    # Compute the mapping
    mapping = map_histogram(source_cdf, target_cdf)

    # Map the pixels in the source image to match the target histogram
    matched_img = np.interp(source_img.ravel(), np.arange(256), mapping).reshape(source_img.shape)
    matched_img = np.uint8(matched_img)
    return matched_img

def plot_images_and_histograms(source_img, target_img, matched_img):
    """Plots the original, reference, and matched images with their histograms."""
    plt.figure(figsize=(12, 12))

    # Original Image
    plt.subplot(3, 2, 1)
    plt.imshow(source_img, cmap='gray')
    plt.title('Source Image')
    plt.axis('off')

    # Histogram of Original Image
    plt.subplot(3, 2, 2)
    plt.hist(source_img.ravel(), bins=256, range=[0,256])
    plt.title('Histogram of Source Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Target Image
    plt.subplot(3, 2, 3)
    plt.imshow(target_img, cmap='gray')
    plt.title('Target Image')
    plt.axis('off')

    # Histogram of Target Image
    plt.subplot(3, 2, 4)
    plt.hist(target_img.ravel(), bins=256, range=[0,256])
    plt.title('Histogram of Target Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Matched Image
    plt.subplot(3, 2, 5)
    plt.imshow(matched_img, cmap='gray')
    plt.title('Matched Image')
    plt.axis('off')

    # Histogram of Matched Image
    plt.subplot(3, 2, 6)
    plt.hist(matched_img.ravel(), bins=256, range=(0,256))
    plt.title('Histogram of Matched Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Load source and target images in grayscale
source_image_path = 'source_image.png'
target_image_path = 'target_image.png'

source_image = cv2.imread(source_image_path)
target_image = cv2.imread(target_image_path)

if source_image is None or target_image is None:
    raise ValueError("One of the images was not found or unable to open.")

# Perform histogram matching
matched_image = histogram_matching(source_image, target_image)

# Plot the images and histograms
plot_images_and_histograms(source_image, target_image, matched_image)

# Optionally save the result
output_path = 'matched_image.jpg'  # Replace with your output path
cv2.imwrite(output_path, matched_image)
