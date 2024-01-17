import cv2
import numpy as np


def my_average_blur(image, kernel_size):
    height, width = image.shape[:2]
    result_image = np.zeros_like(image)

    k_width, k_height = kernel_size
    k_size = k_width * k_height

    for y in range(height):
        for x in range(width):
            roi = image[y:y + k_height, x:x + k_width]
            if roi.shape[:2] == kernel_size:
                result_image[y, x] = np.sum(roi) / k_size

    return result_image


def my_gaussian_blur(image, kernel_size, sigma):
    height, width = image.shape[:2]
    result_image = np.zeros_like(image, dtype=np.float32)

    k_width, k_height = kernel_size
    k_half_width, k_half_height = k_width // 2, k_height // 2

    # Создаем ядро Гаусса
    kernel = np.zeros(kernel_size, dtype=np.float32)

    # Рассчитываем центр ядра
    center_x, center_y = k_half_width, k_half_height

    # Рассчитываем коэффициенты ядра
    coeff_sum = 0
    for y in range(-k_half_height, k_half_height + 1):
        for x in range(-k_half_width, k_half_width + 1):
            exponent = -((x ** 2 + y ** 2) / (2 * sigma ** 2))
            kernel[y + center_y, x + center_x] = np.exp(exponent) / (2 * np.pi * sigma ** 2)
            coeff_sum += kernel[y + center_y, x + center_x]

    # Нормализуем ядро
    kernel /= coeff_sum

    # Применяем фильтр к каждому пикселю
    for y in range(height):
        for x in range(width):
            sum_value = 0
            for ky in range(-k_half_height, k_half_height + 1):
                for kx in range(-k_half_width, k_half_width + 1):
                    pixel_x = x + kx
                    pixel_y = y + ky
                    if 0 <= pixel_x < width and 0 <= pixel_y < height:
                        sum_value += image[pixel_y, pixel_x] * kernel[ky + center_y, kx + center_x]
            result_image[y, x] = sum_value

    return result_image.astype(np.uint8)


def my_median_blur(image, kernel_size):
    height, width = image.shape[:2]
    result_image = np.zeros_like(image)

    k_width, k_height = kernel_size
    k_half = k_width * k_height // 2

    # Применяем фильтр к каждому пикселю
    for y in range(height):
        for x in range(width):
            roi = image[y:y + k_height, x:x + k_width]
            if roi.shape[:2] == kernel_size:
                median_value = np.median(roi)
                result_image[y, x] = median_value

    return result_image


def my_laplacian(image):
    # Создаем фильтр Лапласа
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)

    # Применяем фильтр к изображению
    laplacian = cv2.filter2D(image, -1, kernel)

    # Преобразуем значения к неотрицательным целым числам
    laplacian_abs = np.abs(laplacian).astype(np.uint8)

    return laplacian_abs


def my_sobel(image):
    # Создаем ядра для горизонтального и вертикального фильтров
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    height, width = image.shape[:2]
    sobel_x = np.zeros_like(image, dtype=np.float32)
    sobel_y = np.zeros_like(image, dtype=np.float32)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Применяем свертку
            pixel_sum_x = (
                    kernel_x[0, 0] * image[y - 1, x - 1] + kernel_x[0, 1] * image[y - 1, x] + kernel_x[0, 2] * image[
                y - 1, x + 1] +
                    kernel_x[1, 0] * image[y, x - 1] + kernel_x[1, 1] * image[y, x] + kernel_x[1, 2] * image[y, x + 1] +
                    kernel_x[2, 0] * image[y + 1, x - 1] + kernel_x[2, 1] * image[y + 1, x] + kernel_x[2, 2] * image[
                        y + 1, x + 1]
            )

            pixel_sum_y = (
                    kernel_y[0, 0] * image[y - 1, x - 1] + kernel_y[0, 1] * image[y - 1, x] + kernel_y[0, 2] * image[
                y - 1, x + 1] +
                    kernel_y[1, 0] * image[y, x - 1] + kernel_y[1, 1] * image[y, x] + kernel_y[1, 2] * image[y, x + 1] +
                    kernel_y[2, 0] * image[y + 1, x - 1] + kernel_y[2, 1] * image[y + 1, x] + kernel_y[2, 2] * image[
                        y + 1, x + 1]
            )

            sobel_x[y, x] = pixel_sum_x
            sobel_y[y, x] = pixel_sum_y

    # Преобразуем значения к неотрицательным целым числам
    sobel_x_abs = np.abs(sobel_x).astype(np.uint8)
    sobel_y_abs = np.abs(sobel_y).astype(np.uint8)

    return sobel_x_abs, sobel_y_abs


# Применение моей реализации линейной фильтрации
def my_linear_filter(image, kernel):
    height, width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]

    # Половина размера ядра для корректного применения фильтра
    half_kernel_height = kernel_height // 2
    half_kernel_width = kernel_width // 2

    result = np.zeros_like(image, dtype=np.float32)

    for y in range(half_kernel_height, height - half_kernel_height):
        for x in range(half_kernel_width, width - half_kernel_width):
            # Применяем фильтр к каждому пикселю
            sum = 0
            for ky in range(-half_kernel_height, half_kernel_height + 1):
                for kx in range(-half_kernel_width, half_kernel_width + 1):
                    sum += kernel[ky + half_kernel_height, kx + half_kernel_width] * image[y + ky, x + kx]

            result[y, x] = sum

    # Нормализуем результат к неотрицательным целым числам
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result
