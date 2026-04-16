import cv2
import numpy as np
import matplotlib.pyplot as plt

def binary_algorithm(binary_img, L):
    # compute path lengths
    lambda_plus = compute_lambda_plus(binary_img)
    lambda_minus = compute_lambda_minus(binary_img)

    # total path length through each pixel
    lambda_total = lambda_plus + lambda_minus - 1

    # keep only pixels that belong to paths of length >= L
    result = np.full(binary_img.shape, 255, dtype=np.uint8)
    result[(binary_img == 1) & (lambda_total >= L)] = 0

    return result


def compute_lambda_plus(img):
    h, w = img.shape
    lambda_plus = np.zeros((h, w), dtype=np.int32)

    # from bottom to top
    for i in range(h - 1, -1, -1):
        for j in range(w):
            if img[i, j] == 0:
                continue

            max_val = 0

            ni = i + 1
            if ni < h:
                # bottom-left
                nj = j - 1
                if nj >= 0 and img[ni, nj] == 1:
                    val = lambda_plus[ni, nj]
                    if val > max_val:
                        max_val = val

                # bottom
                nj = j
                if img[ni, nj] == 1:
                    val = lambda_plus[ni, nj]
                    if val > max_val:
                        max_val = val

                # bottom-right
                nj = j + 1
                if nj < w and img[ni, nj] == 1:
                    val = lambda_plus[ni, nj]
                    if val > max_val:
                        max_val = val

            lambda_plus[i, j] = max_val + 1

    return lambda_plus


def compute_lambda_minus(img):
    h, w = img.shape
    lambda_minus = np.zeros((h, w), dtype=np.int32)

    # from top to bottom
    for i in range(h):
        for j in range(w):
            if img[i, j] == 0:
                continue

            max_val = 0

            ni = i - 1
            if ni >= 0:
                # top-left
                nj = j - 1
                if nj >= 0 and img[ni, nj] == 1:
                    val = lambda_minus[ni, nj]
                    if val > max_val:
                        max_val = val

                # top
                nj = j
                if img[ni, nj] == 1:
                    val = lambda_minus[ni, nj]
                    if val > max_val:
                        max_val = val

                # top-right
                nj = j + 1
                if nj < w and img[ni, nj] == 1:
                    val = lambda_minus[ni, nj]
                    if val > max_val:
                        max_val = val

            lambda_minus[i, j] = max_val + 1

    return lambda_minus

if __name__ == "__main__":

    img = cv2.imread("./dataset/9.bmp", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Не вдалося завантажити зображення")
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary01 = (binary == 0).astype(np.uint8)

    #apply algorithm
    result1 = binary_algorithm(binary01, L=10)

    result2 = binary_algorithm(binary01, L=20)

    #plotting
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Зображення з рівнями сірого")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(result1, cmap="gray")
    plt.title("Результат(L = 10)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(binary, cmap="gray")
    plt.title("Бінарне зображення")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(result2, cmap="gray")
    plt.title("Результат(L = 20)")
    plt.axis("off")

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()
