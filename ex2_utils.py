import math

import cv2
import numpy as np


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return np.int_(323825059)


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convoluted array
    """
    cpy = np.copy(in_signal)
    kernel_size = len(k_size)
    signal_size = len(in_signal)
    cpy = np.concatenate((cpy, np.zeros(kernel_size - 1)), axis=0)
    cpy = np.concatenate((np.zeros(kernel_size - 1), cpy), axis=0)
    out = np.zeros(signal_size + kernel_size - 1)
    for idx in range(0, len(out)):
        out[idx] = cpy[idx:idx + kernel_size] @ k_size.T
    return out


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convoluted image
    """
    flipped_kernel = np.flip(kernel)
    kernel_h, kernel_w = flipped_kernel.shape
    image_h, image_w = in_image.shape
    image_pad = np.pad(in_image, (kernel_h // 2, kernel_w // 2), "edge")
    image_conv = np.zeros((image_h, image_w))
    for i in range(0, image_h):
        for j in range(0, image_w):
            image_conv[i, j] = (image_pad[i:i + kernel_h, j:j + kernel_w] * flipped_kernel).sum()
    return image_conv


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Gray scale image
    :return: (directions, magnitude)
    """
    dev = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    dev_x = conv2D(in_image, dev)
    dev_y = conv2D(in_image, dev.transpose())
    mag = np.sqrt(np.power(dev_x, 2) + np.power(dev_y, 2))
    direct = np.arctan(dev_y, dev_x)
    return direct, mag


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    gaussian_kernel = compute_gaussian_kernel(k_size, kernel_sigma(k_size))
    return conv2D(in_image, gaussian_kernel)


def kernel_sigma(kernel_size: int) -> float:
    return 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8


def compute_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size))
    center = (kernel_size - 1) / 2

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            exponent = -(x ** 2 + y ** 2) / (2 * sigma ** 2)
            coefficient = 1 / (2 * np.pi * sigma ** 2)
            kernel[i, j] = coefficient * np.exp(exponent)

    kernel /= np.sum(kernel)
    return kernel


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using Open CV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    kernel = cv2.getGaussianKernel(k_size, kernel_sigma(k_size))
    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    dev_simple_matrix = np.array([[-1, 0, 1], [0, 0, 0], [-1, 0, 1]])
    after_lap_conv = conv2D(img, dev_simple_matrix)
    return search_zero_crossing(after_lap_conv)


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    laplacian_matrix = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    after_LoG_conv = conv2D(blurImage2(img, 5), laplacian_matrix)
    return search_zero_crossing(after_LoG_conv)


def search_zero_crossing(mat: np.ndarray) -> np.ndarray:
    edges = np.zeros_like(mat)
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if mat[i, j] == 0:
                edges[i, j] = 1
                continue
            if i in range(1, mat.shape[0]):
                if (mat[i - 1, j] >= 0 and mat[i, j] < 0) or (mat[i - 1, j] < 0 and mat[i, j] >= 0):
                    edges[i, j] = 1
                    continue
                if i < mat.shape[0] - 1:
                    if (mat[i - 1, j] >= 0 and mat[i + 1, j] < 0) or (mat[i - 1, j] < 0 and mat[i + 1, j] >= 0):
                        edges[i, j] = 1
                        continue

            if j in range(1, mat.shape[1]):
                if (mat[i, j - 1] >= 0 and mat[i, j] < 0) or (mat[i, j - 1] < 0 and mat[i, j] >= 0):
                    edges[i, j] = 1
                    continue
                if j < mat.shape[1] - 1:
                    if (mat[i, j - 1] >= 0 and mat[i, j + 1] < 0) or (mat[i, j - 1] < 0 and mat[i, j + 1] >= 0):
                        edges[i, j] = 1
                        continue

    return edges


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use Open CV function: cv.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
    """
    processed_image = (img * 255).astype(np.uint8)
    edge_detected_image = cv2.Canny(processed_image, 100, 200)
    calculated_points = generate_points(min_radius, max_radius, 100)
    edge_pixels = extract_edge_pixels(edge_detected_image)
    circle_counts = tally_circles(edge_pixels, calculated_points)
    final_result = filter_results(circle_counts, 100, 0.5)
    return final_result


def generate_points(lower_radius, upper_radius, number_of_steps):
    circle_points = []
    for rad in range(lower_radius, upper_radius + 1):
        for step in range(number_of_steps):
            x_cord = int(np.cos(np.pi * (step / number_of_steps) * 2) * rad)
            y_cord = int(np.sin(np.pi * (step / number_of_steps) * 2) * rad)
            circle_points.append((x_cord, y_cord, rad))
    return circle_points


def extract_edge_pixels(edge_detected_image):
    edge_locations = []
    num_rows, num_cols = edge_detected_image.shape
    for x in range(num_rows):
        for y in range(num_cols):
            if edge_detected_image[x, y] == 255:
                edge_locations.append((x, y))
    return edge_locations


def tally_circles(edge_locations, circle_points):
    circle_dictionary = {}
    for x1, y1 in edge_locations:
        for x2, y2, rad in circle_points:
            dx, dy = x1 - x2, y1 - y2
            circle_key = circle_dictionary.get((dy, dx, rad))
            if circle_key is None:
                circle_dictionary[(dy, dx, rad)] = 1
            else:
                circle_dictionary[(dy, dx, rad)] += 1
    return circle_dictionary


def filter_results(circle_dictionary, number_of_steps, threshold_ratio):
    final_result = []
    for circle, count in sorted(circle_dictionary.items(), key=lambda v: -v[1]):
        nx, ny, rad = circle
        if count / number_of_steps >= threshold_ratio and all(
                (nx - x) ** 2 + (ny - y) ** 2 > r ** 2 for x, y, r in final_result):
            final_result.append((nx, ny, rad))
    return final_result


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: Open CV implementation, my implementation
    """
    pass
