import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

MAX_STACK_SIZE = 256

def precompute_stack_path_opening(img):
    if img.dtype != np.uint8:
        raise ValueError("img must be a uint8 grayscale image")

    work = 255 - img
    mask = (work > 0).astype(np.uint8)

    lambda_plus_set = update_lambda_plus_set(work, mask)
    lambda_minus_set = update_lambda_minus_set(work, mask)

    return work, mask, lambda_plus_set, lambda_minus_set


def stack_path_opening_from_precomputed(work, mask, lambda_plus_set, lambda_minus_set, L):
    result = build_result(
        lambda_plus_set[0], lambda_plus_set[1], lambda_plus_set[2],
        lambda_minus_set[0], lambda_minus_set[1], lambda_minus_set[2],
        mask, L
    )
    return 255 - result


@njit
def merge(pred_levels, pred_lambdas, pred_sizes, pred_count):
    merged_levels = np.empty(MAX_STACK_SIZE, dtype=np.uint8)
    merged_lambdas = np.zeros(MAX_STACK_SIZE, dtype=np.int32)
    merged_size = 0

    for p in range(pred_count):
        size = pred_sizes[p]

        for k in range(size):
            level = pred_levels[p, k]
            lmbda = pred_lambdas[p, k]

            found = -1
            for m in range(merged_size):
                if merged_levels[m] == level:
                    found = m
                    break

            if found == -1:
                merged_levels[merged_size] = level
                merged_lambdas[merged_size] = lmbda
                merged_size += 1
            else:
                if lmbda > merged_lambdas[found]:
                    merged_lambdas[found] = lmbda

    for i in range(1, merged_size):
        key_level = merged_levels[i]
        key_lambda = merged_lambdas[i]
        j = i - 1

        while j >= 0 and merged_levels[j] > key_level:
            merged_levels[j + 1] = merged_levels[j]
            merged_lambdas[j + 1] = merged_lambdas[j]
            j -= 1

        merged_levels[j + 1] = key_level
        merged_lambdas[j + 1] = key_lambda

    return merged_levels, merged_lambdas, merged_size


def update_lambda_plus_set(img, mask):
    return _update_lambda_plus_set(img, mask)


def update_lambda_minus_set(img, mask):
    return _update_lambda_minus_set(img, mask)


@njit
def _update_lambda_plus_set(img, mask):
    h, w = img.shape

    levels = np.zeros((h, w, MAX_STACK_SIZE), dtype=np.uint8)
    lambdas = np.zeros((h, w, MAX_STACK_SIZE), dtype=np.int32)
    sizes = np.zeros((h, w), dtype=np.int32)

    pred_levels = np.empty((3, MAX_STACK_SIZE), dtype=np.uint8)
    pred_lambdas = np.empty((3, MAX_STACK_SIZE), dtype=np.int32)
    pred_sizes = np.zeros(3, dtype=np.int32)

    for x in range(h - 1, -1, -1):
        for y in range(w):
            if mask[x, y] == 0:
                continue

            value = img[x, y]
            pred_count = 0

            ni = x + 1
            if ni < h:
                for nj in (y - 1, y, y + 1):
                    if 0 <= nj < w and mask[ni, nj] == 1:
                        size = sizes[ni, nj]
                        pred_sizes[pred_count] = size

                        for k in range(size):
                            pred_levels[pred_count, k] = levels[ni, nj, k]
                            pred_lambdas[pred_count, k] = lambdas[ni, nj, k]

                        pred_count += 1

            merged_levels, merged_lambdas, merged_size = merge(
                pred_levels, pred_lambdas, pred_sizes, pred_count
            )

            max_len = 0
            for k in range(merged_size):
                l = merged_levels[k]
                lmbda = merged_lambdas[k]

                if l >= value and lmbda > max_len:
                    max_len = lmbda

            lambda_plus_temp = max_len + 1
            current_size = 0

            for k in range(merged_size):
                l = merged_levels[k]
                lmbda = merged_lambdas[k]

                if l < value:
                    levels[x, y, current_size] = l
                    lambdas[x, y, current_size] = lmbda + 1
                    current_size += 1

            levels[x, y, current_size] = value
            lambdas[x, y, current_size] = lambda_plus_temp
            current_size += 1

            sizes[x, y] = current_size

    return levels, lambdas, sizes


@njit
def _update_lambda_minus_set(img, mask):
    h, w = img.shape

    levels = np.zeros((h, w, MAX_STACK_SIZE), dtype=np.uint8)
    lambdas = np.zeros((h, w, MAX_STACK_SIZE), dtype=np.int32)
    sizes = np.zeros((h, w), dtype=np.int32)

    pred_levels = np.empty((3, MAX_STACK_SIZE), dtype=np.uint8)
    pred_lambdas = np.empty((3, MAX_STACK_SIZE), dtype=np.int32)
    pred_sizes = np.zeros(3, dtype=np.int32)

    for x in range(h):
        for y in range(w):
            if mask[x, y] == 0:
                continue

            value = img[x, y]
            pred_count = 0

            ni = x - 1
            if ni >= 0:
                for nj in (y - 1, y, y + 1):
                    if 0 <= nj < w and mask[ni, nj] == 1:
                        size = sizes[ni, nj]
                        pred_sizes[pred_count] = size

                        for k in range(size):
                            pred_levels[pred_count, k] = levels[ni, nj, k]
                            pred_lambdas[pred_count, k] = lambdas[ni, nj, k]

                        pred_count += 1

            merged_levels, merged_lambdas, merged_size = merge(
                pred_levels, pred_lambdas, pred_sizes, pred_count
            )

            max_len = 0
            for k in range(merged_size):
                l = merged_levels[k]
                lmbda = merged_lambdas[k]

                if l >= value and lmbda > max_len:
                    max_len = lmbda

            lambda_minus_temp = max_len + 1
            current_size = 0

            for k in range(merged_size):
                l = merged_levels[k]
                lmbda = merged_lambdas[k]

                if l < value:
                    levels[x, y, current_size] = l
                    lambdas[x, y, current_size] = lmbda + 1
                    current_size += 1

            levels[x, y, current_size] = value
            lambdas[x, y, current_size] = lambda_minus_temp
            current_size += 1

            sizes[x, y] = current_size

    return levels, lambdas, sizes


@njit
def build_result(
    plus_levels, plus_lambdas, plus_sizes,
    minus_levels, minus_lambdas, minus_sizes,
    mask, L
):
    h, w = mask.shape
    result = np.zeros((h, w), dtype=np.uint8)

    for x in range(h):
        for y in range(w):
            if mask[x, y] == 0:
                continue

            max_valid_level = 0

            i = 0
            j = 0
            plus_size = plus_sizes[x, y]
            minus_size = minus_sizes[x, y]

            while i < plus_size and j < minus_size:
                level_plus = plus_levels[x, y, i]
                level_minus = minus_levels[x, y, j]

                if level_plus == level_minus:
                    total_length = plus_lambdas[x, y, i] + minus_lambdas[x, y, j] - 1

                    if total_length >= L and level_plus > max_valid_level:
                        max_valid_level = level_plus

                    i += 1
                    j += 1
                elif level_plus < level_minus:
                    i += 1
                else:
                    j += 1

            result[x, y] = np.uint8(max_valid_level)

    return result


if __name__ == "__main__":
    img = cv2.imread("./dataset/2.bmp", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Не вдалося завантажити зображення")

    work, mask, lambda_plus_set, lambda_minus_set = precompute_stack_path_opening(img)

    result1 = stack_path_opening_from_precomputed(work, mask, lambda_plus_set, lambda_minus_set, L=10)
    result2 = stack_path_opening_from_precomputed(work, mask, lambda_plus_set, lambda_minus_set, L=20)
    result3 = stack_path_opening_from_precomputed(work, mask, lambda_plus_set, lambda_minus_set, L=30)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Зображення з рівнями сірого")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(result1, cmap="gray")
    plt.title("Шляхове розкриття на основі стеку (L = 10)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(result2, cmap="gray")
    plt.title("Шляхове розкриття на основі стеку (L = 20)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(result3, cmap="gray")
    plt.title("Шляхове розкриття на основі стеку (L = 30)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()