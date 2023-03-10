import cv2
import numpy as np
import matplotlib.pyplot as plt


def run():
    image = cv2.imread('coin_error.png')
    auto_result = brightness_correction(image)
    cv2.imshow('auto_result', auto_result)
    cv2.imshow('image', image)
    correction_result = noise_correction(auto_result)
    cv2.imshow("After noise correction", correction_result)
    cv2.imwrite("Corrected_image_v1.png", correction_result)
    # sharp_image = sharpen_image(correction_result)
    # cv2.imshow("Sharpen Image", sharp_image)
    # cv2.imwrite("Corrected_image.png", sharp_image)

    cv2.waitKey()


def run_v2():
    image = cv2.imread('coin_error.png')
    cv2.imshow("original", image)

    correction_result = noise_correction(image)
    cv2.imshow("After noise correction", correction_result)
    cv2.imwrite("Corrected_image_v1.png", correction_result)

    # auto brightness
    auto_result = brightness_correction(correction_result)
    cv2.imshow('auto_result', auto_result)

    cv2.waitKey()


def brightness_correction(image: str, clip_hist_percent=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size: int = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = [float(hist[0])]
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum: float = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray: int = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha: float = 255 / (maximum_gray - minimum_gray)
    beta: float = -minimum_gray * alpha

    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray], [0], None, [256], [minimum_gray, maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0, 256])
    plt.show()

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result


def noise_correction(image):
    height, width = image.shape[:2]
    dst: cv2.Mat = image
    window: list[9] = []

    for y in range(height):
        for x in range(width):
            dst[y][x] = 0.0

    for y in range(height):
        for x in range(width):
            if x != 0 and x < 1919 and y != 0 and y < 1047:

            # sort the window to find median
            # cv2.insertionSort(window)

            # assign the median to centered element of the matrix
            dst[y][x] = window[4]

    cv2.imshow("iets", dst)
    cv2.waitKey()

    img_without_noise = cv2.medianBlur(image, 3)
    return img_without_noise


def insertionSort():
    x, y: int = 0
    window: list[9] = []

    for i in range(x):
        temp = window[i]
        for j in range(y):





def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return image_sharp


if __name__ == "__main__":
    run_v2()
