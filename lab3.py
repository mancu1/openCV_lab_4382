import cv2
import numpy as np


def ideal_lowpass_filter(image, D0):
    # Применяем прямое дискретное преобразование Фурье к изображению
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Определяем размеры изображения и центр частотной области
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Создаем фильтр
    H = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if distance <= D0:
                H[i, j] = 1

    # Умножаем фурье-образ изображения на фильтр
    G_transform_shifted = f_transform_shifted * H

    # Вычисляем обратное преобразование Фурье
    G_transform = np.fft.ifftshift(G_transform_shifted)
    filtered_image = np.fft.ifft2(G_transform).real

    return filtered_image


def butterworth_lowpass_filter(image, D0, n):
    # Применяем прямое дискретное преобразование Фурье к изображению
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Определяем размеры изображения и центр частотной области
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Создаем фильтр Баттерворта
    H = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            H[i, j] = 1 / (1 + (distance / D0) ** (2 * n))

    # Умножаем фурье-образ изображения на фильтр
    G_transform_shifted = f_transform_shifted * H

    # Вычисляем обратное преобразование Фурье
    G_transform = np.fft.ifftshift(G_transform_shifted)
    filtered_image = np.fft.ifft2(G_transform).real

    return filtered_image


def gaussian_lowpass_filter(image, D0, sigma):
    # Применяем прямое дискретное преобразование Фурье к изображению
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Определяем размеры изображения и центр частотной области
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Создаем фильтр Гаусса
    H = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            H[i, j] = np.exp(-(distance ** 2) / (2 * sigma ** 2))

    # Умножаем фурье-образ изображения на фильтр
    G_transform_shifted = f_transform_shifted * H

    # Вычисляем обратное преобразование Фурье
    G_transform = np.fft.ifftshift(G_transform_shifted)
    filtered_image = np.fft.ifft2(G_transform).real

    return filtered_image





def ideal_bandpass_filter(image, D0_low, D0_high):
    # Применяем прямое дискретное преобразование Фурье к изображению
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Определяем размеры изображения и центр частотной области
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Создаем фильтр
    H = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if D0_low <= distance <= D0_high:
                H[i, j] = 1

    # Умножаем фурье-образ изображения на фильтр
    G_transform_shifted = f_transform_shifted * H

    # Вычисляем обратное преобразование Фурье
    G_transform = np.fft.ifftshift(G_transform_shifted)
    filtered_image = np.fft.ifft2(G_transform).real

    return filtered_image





def butterworth_bandpass_filter(image, D0_low, D0_high, n):
    # Применяем прямое дискретное преобразование Фурье к изображению
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Определяем размеры изображения и центр частотной области
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Создаем фильтр Баттерворта
    H = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            H[i, j] = 1 / (1 + ((distance ** 2) / (D0_low * D0_high)) ** n)

    # Умножаем фурье-образ изображения на фильтр
    G_transform_shifted = f_transform_shifted * H

    # Вычисляем обратное преобразование Фурье
    G_transform = np.fft.ifftshift(G_transform_shifted)
    filtered_image = np.fft.ifft2(G_transform).real

    return filtered_image





def gaussian_bandpass_filter(image, D0_low, D0_high, sigma):
    # Применяем прямое дискретное преобразование Фурье к изображению
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Определяем размеры изображения и центр частотной области
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Создаем фильтр Гаусса
    H = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            H[i, j] = np.exp(-((distance ** 2) - D0_low ** 2) / (2 * sigma ** 2)) * (
                        1 - np.exp(-((distance ** 2) - D0_high ** 2) / (2 * sigma ** 2)))

    # Умножаем фурье-образ изображения на фильтр
    G_transform_shifted = f_transform_shifted * H

    # Вычисляем обратное преобразование Фурье
    G_transform = np.fft.ifftshift(G_transform_shifted)
    filtered_image = np.fft.ifft2(G_transform).real

    return filtered_image





def ideal_highpass_filter(image, D0):
    # Применяем прямое дискретное преобразование Фурье к изображению
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Определяем размеры изображения и центр частотной области
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Создаем фильтр
    H = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            if distance > D0:
                H[i, j] = 1

    # Умножаем фурье-образ изображения на фильтр
    G_transform_shifted = f_transform_shifted * H

    # Вычисляем обратное преобразование Фурье
    G_transform = np.fft.ifftshift(G_transform_shifted)
    filtered_image = np.fft.ifft2(G_transform).real

    return filtered_image





def butterworth_highpass_filter(image, D0, n):
    # Применяем прямое дискретное преобразование Фурье к изображению
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Определяем размеры изображения и центр частотной области
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Создаем фильтр Баттерворта
    H = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            H[i, j] = 1 / (1 + (D0 / distance) ** (2 * n))

    # Умножаем фурье-образ изображения на фильтр
    G_transform_shifted = f_transform_shifted * H

    # Вычисляем обратное преобразование Фурье
    G_transform = np.fft.ifftshift(G_transform_shifted)
    filtered_image = np.fft.ifft2(G_transform).real

    return filtered_image





def gaussian_highpass_filter(image, D0, sigma):
    # Применяем прямое дискретное преобразование Фурье к изображению
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Определяем размеры изображения и центр частотной области
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # Создаем фильтр Гаусса
    H = np.zeros_like(image, dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row) ** 2 + (j - center_col) ** 2)
            H[i, j] = 1 - np.exp(-((distance ** 2) / (2 * sigma ** 2)))

    # Умножаем фурье-образ изображения на фильтр
    G_transform_shifted = f_transform_shifted * H

    # Вычисляем обратное преобразование Фурье
    G_transform = np.fft.ifftshift(G_transform_shifted)
    filtered_image = np.fft.ifft2(G_transform).real

    return filtered_image
