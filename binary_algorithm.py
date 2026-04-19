import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_predecessors(img, i, j):
    h, w = img.shape
    predecessors = []

    ni = i + 1
    if ni < h:
        for nj in (j - 1, j, j + 1):
            if 0 <= nj < w and img[ni, nj] == 1:
                predecessors.append((ni, nj))

    return predecessors


def get_successors(img, i, j):
    h, w = img.shape
    successors = []

    ni = i - 1
    if ni >= 0:
        for nj in (j - 1, j, j + 1):
            if 0 <= nj < w and img[ni, nj] == 1:
                successors.append((ni, nj))

    return successors


def compute_lambda_plus(img):
    h, w = img.shape
    lambda_plus = np.zeros((h, w), dtype=np.int32)

    # обхід знизу вгору
    for i in range(h - 1, -1, -1):
        for j in range(w):
            if img[i, j] == 0:
                continue

            predecessors = get_predecessors(img, i, j)

            max_val = 0
            for pi, pj in predecessors:
                val = lambda_plus[pi, pj]
                if val > max_val:
                    max_val = val

            lambda_plus[i, j] = max_val + 1

    return lambda_plus


def compute_lambda_minus(img):
    h, w = img.shape
    lambda_minus = np.zeros((h, w), dtype=np.int32)

    # обхід зверху вниз
    for i in range(h):
        for j in range(w):
            if img[i, j] == 0:
                continue

            successors = get_successors(img, i, j)

            max_val = 0
            for si, sj in successors:
                val = lambda_minus[si, sj]
                if val > max_val:
                    max_val = val

            lambda_minus[i, j] = max_val + 1

    return lambda_minus


def binary_algorithm(binary_img, L):
    # обчислення довжин шляху
    lambda_plus = compute_lambda_plus(binary_img)
    lambda_minus = compute_lambda_minus(binary_img)

    # повна довжина шляху через піксель
    lambda_total = lambda_plus + lambda_minus - 1

    # залишаємо лише ті пікселі, що належать шляхам довжини >= L
    result = np.full(binary_img.shape, 255, dtype=np.uint8)
    result[(binary_img == 1) & (lambda_total >= L)] = 0

    return result


if __name__ == "__main__":
    img = cv2.imread("./dataset/9.bmp", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Не вдалося завантажити зображення")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary01 = (binary == 0).astype(np.uint8)

    # застосування алгоритму
    result1 = binary_algorithm(binary01, L=10)
    result2 = binary_algorithm(binary01, L=20)

    # візуалізація
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Зображення з рівнями сірого")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(result1, cmap="gray")
    plt.title("Результат (L = 10)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(binary, cmap="gray")
    plt.title("Бінарне зображення")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(result2, cmap="gray")
    plt.title("Результат (L = 20)")
    plt.axis("off")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()