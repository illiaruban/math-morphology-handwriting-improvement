import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

MAX_STACK_SIZE = 256
MAX_NEIGHBORS = 4


def handwriting_enhancement_algorithm(img, L, darken_value=60, blur=True):
    if img.dtype != np.uint8:
        raise ValueError("img must be a uint8 grayscale image")

    if blur:
        img = cv2.medianBlur(img, 3)

    work = 255 - img
    path_mask = (work > 0).astype(np.uint8)

    lambda_plus_set = update_lambda_plus_set(work, path_mask)
    lambda_minus_set = update_lambda_minus_set(work, path_mask)

    raw_result = build_result(
        lambda_plus_set[0], lambda_plus_set[1], lambda_plus_set[2],
        lambda_minus_set[0], lambda_minus_set[1], lambda_minus_set[2],
        path_mask,
        L
    )

    stack_path_result = 255 - raw_result

    confirmed_mask = compute_mask(stack_path_result)

    result = enhance_with_mask(
        original_img=img,
        mask=confirmed_mask,
        darken_value=darken_value
    )

    return result


def compute_mask(result_img):
    confirmed = cv2.adaptiveThreshold(
        result_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    return (confirmed > 0).astype(np.uint8)


def enhance_with_mask(original_img, mask, darken_value=60):
    img_float = original_img.astype(np.float32)
    mask_float = mask.astype(np.float32)

    enhanced = img_float - mask_float * darken_value

    return np.clip(enhanced, 0, 255).astype(np.uint8)


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


@njit
def update_lambda_plus_set(img, mask):
    h, w = img.shape

    levels = np.zeros((h, w, MAX_STACK_SIZE), dtype=np.uint8)
    lambdas = np.zeros((h, w, MAX_STACK_SIZE), dtype=np.int32)
    sizes = np.zeros((h, w), dtype=np.int32)

    pred_levels = np.empty((MAX_NEIGHBORS, MAX_STACK_SIZE), dtype=np.uint8)
    pred_lambdas = np.empty((MAX_NEIGHBORS, MAX_STACK_SIZE), dtype=np.int32)
    pred_sizes = np.zeros(MAX_NEIGHBORS, dtype=np.int32)

    for x in range(h):
        for y in range(w - 1, -1, -1):
            if mask[x, y] == 0:
                continue

            value = img[x, y]
            pred_count = 0

            neighbors = (
                (x - 1, y - 1),
                (x - 1, y),
                (x - 1, y + 1),
                (x, y + 1)
            )

            for n in range(MAX_NEIGHBORS):
                ni = neighbors[n][0]
                nj = neighbors[n][1]

                if 0 <= ni < h and 0 <= nj < w and mask[ni, nj] == 1:
                    size = sizes[ni, nj]
                    pred_sizes[pred_count] = size

                    for k in range(size):
                        pred_levels[pred_count, k] = levels[ni, nj, k]
                        pred_lambdas[pred_count, k] = lambdas[ni, nj, k]

                    pred_count += 1

            merged_levels, merged_lambdas, merged_size = merge(
                pred_levels,
                pred_lambdas,
                pred_sizes,
                pred_count
            )

            max_len = 0

            for k in range(merged_size):
                level = merged_levels[k]
                lmbda = merged_lambdas[k]

                if level >= value and lmbda > max_len:
                    max_len = lmbda

            lambda_plus_temp = max_len + 1
            current_size = 0

            for k in range(merged_size):
                level = merged_levels[k]
                lmbda = merged_lambdas[k]

                if level < value:
                    levels[x, y, current_size] = level
                    lambdas[x, y, current_size] = lmbda + 1
                    current_size += 1

            levels[x, y, current_size] = value
            lambdas[x, y, current_size] = lambda_plus_temp
            current_size += 1

            sizes[x, y] = current_size

    return levels, lambdas, sizes


@njit
def update_lambda_minus_set(img, mask):
    h, w = img.shape

    levels = np.zeros((h, w, MAX_STACK_SIZE), dtype=np.uint8)
    lambdas = np.zeros((h, w, MAX_STACK_SIZE), dtype=np.int32)
    sizes = np.zeros((h, w), dtype=np.int32)

    pred_levels = np.empty((MAX_NEIGHBORS, MAX_STACK_SIZE), dtype=np.uint8)
    pred_lambdas = np.empty((MAX_NEIGHBORS, MAX_STACK_SIZE), dtype=np.int32)
    pred_sizes = np.zeros(MAX_NEIGHBORS, dtype=np.int32)

    for x in range(h - 1, -1, -1):
        for y in range(w):
            if mask[x, y] == 0:
                continue

            value = img[x, y]
            pred_count = 0

            neighbors = (
                (x, y - 1),
                (x + 1, y - 1),
                (x + 1, y),
                (x + 1, y + 1)
            )

            for n in range(MAX_NEIGHBORS):
                ni = neighbors[n][0]
                nj = neighbors[n][1]

                if 0 <= ni < h and 0 <= nj < w and mask[ni, nj] == 1:
                    size = sizes[ni, nj]
                    pred_sizes[pred_count] = size

                    for k in range(size):
                        pred_levels[pred_count, k] = levels[ni, nj, k]
                        pred_lambdas[pred_count, k] = lambdas[ni, nj, k]

                    pred_count += 1

            merged_levels, merged_lambdas, merged_size = merge(
                pred_levels,
                pred_lambdas,
                pred_sizes,
                pred_count
            )

            max_len = 0

            for k in range(merged_size):
                level = merged_levels[k]
                lmbda = merged_lambdas[k]

                if level >= value and lmbda > max_len:
                    max_len = lmbda

            lambda_minus_temp = max_len + 1
            current_size = 0

            for k in range(merged_size):
                level = merged_levels[k]
                lmbda = merged_lambdas[k]

                if level < value:
                    levels[x, y, current_size] = level
                    lambdas[x, y, current_size] = lmbda + 1
                    current_size += 1

            levels[x, y, current_size] = value
            lambdas[x, y, current_size] = lambda_minus_temp
            current_size += 1

            sizes[x, y] = current_size

    return levels, lambdas, sizes


@njit
def build_result(
    plus_levels,
    plus_lambdas,
    plus_sizes,
    minus_levels,
    minus_lambdas,
    minus_sizes,
    mask,
    L
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
                    total_length = (
                        plus_lambdas[x, y, i]
                        + minus_lambdas[x, y, j]
                        - 1
                    )

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

    result = handwriting_enhancement_algorithm(
        img,
        L=30,
        darken_value=40
    )

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Зображення з рівнями сірого")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap="gray")
    plt.title("Результат")
    plt.axis("off")

    plt.tight_layout()
    plt.show()