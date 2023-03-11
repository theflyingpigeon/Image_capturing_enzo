import cv2
import numpy as np
from PIL import Image, ImageFilter


def run():
    image = cv2.imread('coin_error.png')
    cv2.imshow("original", image)

    correction_result = median_blur(image)
    cv2.imshow("After noise correction", correction_result)
    cv2.imwrite("Corrected_image_v1.png", correction_result)

    # auto brightness
    auto_result = auto_brightness(correction_result, 64)
    cv2.imshow('after auto brightness', auto_result)

    sharpen = sharpen_image(auto_result)
    cv2.imshow("end result", sharpen)

    cv2.waitKey()


def auto_brightness(img, target_mean):
    # Convert image to grayscale
    cv2.imwrite("after_denoise.png", img)
    gray_image = Image.open("after_denoise.png").convert('L')
    # Convert image to numpy array
    array = np.array(gray_image)
    # Compute current mean brightness
    current_mean = np.mean(array)
    # Compute scaling factor
    scale_factor = target_mean / current_mean
    # Apply scaling
    scaled_array = np.clip(array * scale_factor, 0, 255).astype(np.uint8)
    # Convert back to PIL image
    output = Image.fromarray(scaled_array)
    # Convert to RGB if necessary
    # if image.mode == 'RGB':
    #     output = output.convert('RGB')
    open_cv_image = np.array(output)
    return open_cv_image


def median_blur(image, kernel_size=3):
    # Get image dimensions
    height, width = image.shape[:2]
    # Create output image
    output = np.zeros((height, width, 3), dtype=np.uint8)
    # Get kernel radius
    kernel_radius = kernel_size // 2
    # Iterate over image pixels
    for y in range(height):
        for x in range(width):
            # Get kernel indices
            ymin = max(y - kernel_radius, 0)
            ymax = min(y + kernel_radius + 1, height)
            xmin = max(x - kernel_radius, 0)
            xmax = min(x + kernel_radius + 1, width)
            # Extract kernel
            kernel = image[ymin:ymax, xmin:xmax, :]
            # Flatten kernel
            kernel_flat = kernel.reshape(-1, 3)
            # Get median of each channel
            median = np.median(kernel_flat, axis=0)
            # Set output pixel to median
            output[y, x, :] = median.astype(np.uint8)
    return output


def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp


if __name__ == "__main__":
    run()
