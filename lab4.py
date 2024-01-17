import numpy as np
from matplotlib import pyplot as plt


def haar_wavelet_transform(image, levels):
    transformed_image = image.copy()
    rows, cols = image.shape

    for level in range(levels):
        for i in range(0, rows, 2):
            for j in range(0, cols, 2):
                avg = (transformed_image[i, j] + transformed_image[i, j + 1] +
                       transformed_image[i + 1, j] + transformed_image[i + 1, j + 1]) / 4

                h_diff = transformed_image[i, j] - avg
                v_diff = transformed_image[i + 1, j] - avg
                d_diff = transformed_image[i, j + 1] - avg

                transformed_image[i, j] = avg
                transformed_image[i, j + 1] = h_diff
                transformed_image[i + 1, j] = v_diff
                transformed_image[i + 1, j + 1] = d_diff

        rows //= 2
        cols //= 2
        transformed_image = transformed_image[:rows, :cols]

    return transformed_image

def hard_threshold(image, threshold):
    thresholded_image = image.copy()
    thresholded_image[np.abs(image) < threshold] = 0
    return thresholded_image

def soft_threshold(image, threshold):
    thresholded_image = image.copy()
    thresholded_image[np.abs(image) < threshold] = 0
    return np.sign(image) * (np.abs(thresholded_image) - threshold)


def process_image(image):
    transformed_image = haar_wavelet_transform(image, levels=2)

    # Применение жесткого порога
    hard_thresholded_image = hard_threshold(transformed_image, threshold=10)

    # Применение мягкого порога
    soft_thresholded_image = soft_threshold(transformed_image, threshold=10)

    # Вывод изображений
    plt.figure(figsize=(12, 6))
    plt.subplot(221)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(222)
    plt.title('Hear wavelet')
    plt.imshow(transformed_image, cmap='gray')

    plt.subplot(223)
    plt.title('Hard Thresholded Image')
    plt.imshow(hard_thresholded_image, cmap='gray')

    plt.subplot(224)
    plt.title('Soft Thresholded Image')
    plt.imshow(soft_thresholded_image, cmap='gray')

    plt.tight_layout()
    plt.show()
