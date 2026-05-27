import cv2
import numpy as np
import matplotlib.pyplot as plt


def otsu_binarization(img):
    if img.dtype != np.uint8:
        raise ValueError("img must be a uint8 grayscale image")

    _, result = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result


def niblack_binarization(img, window_size=31, k=-0.2):
    if img.dtype != np.uint8:
        raise ValueError("img must be a uint8 grayscale image")

    img = img.astype(np.float32)

    mean = cv2.boxFilter(
        img,
        ddepth=-1,
        ksize=(window_size, window_size),
        normalize=True
    )

    mean_sq = cv2.boxFilter(
        img * img,
        ddepth=-1,
        ksize=(window_size, window_size),
        normalize=True
    )

    variance = mean_sq - mean * mean
    variance = np.maximum(variance, 0) 
    std = np.sqrt(variance)

    threshold = mean + k * std

    result = np.where(img > threshold, 255, 0).astype(np.uint8)

    return result


if __name__ == "__main__":

    img = cv2.imread("./dataset/5.bmp", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Не вдалося завантажити зображення")

    otsu = otsu_binarization(img)
    niblack = niblack_binarization(img, window_size=31, k=-0.2)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Зображення з рівнями сірого")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(otsu, cmap="gray")
    plt.title("Метод Оцу - глобальний")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    plt.imshow(niblack, cmap="gray")
    plt.title("Метод Ніблека - локальний")
    plt.axis("off")

    plt.tight_layout()
    plt.show()