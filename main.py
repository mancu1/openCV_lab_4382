import cv2
import numpy as np
import matplotlib.pyplot as plt

from lab1 import draw_triangle, select_and_save_area, merge_images
from lab2 import my_average_blur, my_gaussian_blur, my_median_blur, my_laplacian, my_sobel, my_linear_filter
from lab3 import ideal_lowpass_filter, butterworth_lowpass_filter, gaussian_lowpass_filter, gaussian_bandpass_filter, \
    gaussian_highpass_filter, butterworth_bandpass_filter, butterworth_highpass_filter, ideal_bandpass_filter, \
    ideal_highpass_filter
from lab4 import process_image


def lab1():
    draw_triangle()
    select_and_save_area()
    merge_images()


def lab2():
    image = cv2.imread("selected_area.png", cv2.IMREAD_GRAYSCALE)

    # Размер ядра для фильтров
    kernel_size = (5, 5)

    # Стандартный фильтр усреднения в OpenCV
    cv2_average_blur = cv2.blur(image, kernel_size)

    # Стандартный гауссов фильтр в OpenCV
    cv2_gaussian_blur = cv2.GaussianBlur(image, kernel_size, sigmaX=2)

    # Стандартный медианный фильтр в OpenCV
    cv2_median_blur = cv2.medianBlur(image, 5)

    # Стандартный лапласиан в OpenCV
    cv2_laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Стандартный оператор Собеля в OpenCV
    cv2_sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    cv2_sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Применение вашей реализации линейной фильтрации
    custom_kernel = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]], dtype=np.float32) / 9
    my_linear_filtered = my_linear_filter(image, custom_kernel)

    # Сравнение результатов
    cv2.imshow("Original Image", image)
    cv2.imshow("My Average Blur", my_average_blur(image, kernel_size))
    cv2.imshow("OpenCV Average Blur", cv2_average_blur)
    cv2.imshow("My Gaussian Blur", my_gaussian_blur(image, kernel_size, sigma=2.0))
    cv2.imshow("OpenCV Gaussian Blur", cv2_gaussian_blur)
    cv2.imshow("My Median Blur", my_median_blur(image, kernel_size))
    cv2.imshow("OpenCV Median Blur", cv2_median_blur)
    cv2.imshow("My Laplacian", my_laplacian(image))
    cv2.imshow("OpenCV Laplacian", cv2.convertScaleAbs(cv2_laplacian))
    cv2.imshow("My Sobel X", my_sobel(image)[0])
    cv2.imshow("OpenCV Sobel X", cv2.convertScaleAbs(cv2_sobel_x))
    cv2.imshow("My Sobel Y", my_sobel(image)[1])
    cv2.imshow("OpenCV Sobel Y", cv2.convertScaleAbs(cv2_sobel_y))
    cv2.imshow("My Linear Filter", my_linear_filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lab3():
    image = cv2.imread('selected_area.png', cv2.IMREAD_GRAYSCALE)

    # Параметры фильтров
    D0_low = 30
    D0_high = 50
    n = 2
    sigma = 10

    # Применить идеальный низкочастотный фильтр
    filtered_ideal_lowpass = ideal_lowpass_filter(image, D0_low)

    # Применить фильтр Баттерворта низкочастотный
    filtered_butterworth_lowpass = butterworth_lowpass_filter(image, D0_low, n)

    # Применить гауссов низкочастотный фильтр
    filtered_gaussian_lowpass = gaussian_lowpass_filter(image, D0_low, sigma)

    # Применить идеальный полосовой фильтр
    filtered_ideal_bandpass = ideal_bandpass_filter(image, D0_low, D0_high)

    # Применить фильтр Баттерворта полосовой
    filtered_butterworth_bandpass = butterworth_bandpass_filter(image, D0_low, D0_high, n)

    # Применить гауссов полосовой фильтр
    filtered_gaussian_bandpass = gaussian_bandpass_filter(image, D0_low, D0_high, sigma)

    # Применить идеальный высокочастотный фильтр
    filtered_ideal_highpass = ideal_highpass_filter(image, D0_high)

    # Применить фильтр Баттерворта высокочастотный
    filtered_butterworth_highpass = butterworth_highpass_filter(image, D0_high, n)

    # Применить гауссов высокочастотный фильтр
    filtered_gaussian_highpass = gaussian_highpass_filter(image, D0_high, sigma)

    # Отобразить результаты фильтрации
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 3, 1), plt.imshow(filtered_ideal_lowpass, cmap='gray'), plt.title('Ideal Lowpass Filter')
    plt.subplot(3, 3, 2), plt.imshow(filtered_butterworth_lowpass, cmap='gray'), plt.title('Butterworth Lowpass Filter')
    plt.subplot(3, 3, 3), plt.imshow(filtered_gaussian_lowpass, cmap='gray'), plt.title('Gaussian Lowpass Filter')
    plt.subplot(3, 3, 4), plt.imshow(filtered_ideal_bandpass, cmap='gray'), plt.title('Ideal Bandpass Filter')
    plt.subplot(3, 3, 5), plt.imshow(filtered_butterworth_bandpass, cmap='gray'), plt.title(
        'Butterworth Bandpass Filter')
    plt.subplot(3, 3, 6), plt.imshow(filtered_gaussian_bandpass, cmap='gray'), plt.title('Gaussian Bandpass Filter')
    plt.subplot(3, 3, 7), plt.imshow(filtered_ideal_highpass, cmap='gray'), plt.title('Ideal Highpass Filter')
    plt.subplot(3, 3, 8), plt.imshow(filtered_butterworth_highpass, cmap='gray'), plt.title(
        'Butterworth Highpass Filter')
    plt.subplot(3, 3, 9), plt.imshow(filtered_gaussian_highpass, cmap='gray'), plt.title('Gaussian Highpass Filter')

    plt.show()


def lab4():
    image = cv2.imread('leaf.jpg', cv2.IMREAD_GRAYSCALE)
    process_image(image)


def main():
    # Вызов функций для каждой задачи lab1
    # lab1()
    # lab2()
    # lab3()
    lab4()


if __name__ == "__main__":
    main()
