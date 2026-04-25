import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

MAX_STACK_SIZE = 256


def handwriting_enhancement_algorithm(img, L, levels_per_mask=30):
    if img.dtype != np.uint8:
        raise ValueError("img must be a uint8 grayscale image")

    processed_img = cv2.medianBlur(img, 3)

    work = 255 - processed_img
    path_mask = (work > 0).astype(np.uint8)

    lambda_plus_set = update_lambda_plus_set(work, path_mask)
    lambda_minus_set = update_lambda_minus_set(work, path_mask)

    raw_result = build_result(
        lambda_plus_set[0], lambda_plus_set[1], lambda_plus_set[2],
        lambda_minus_set[0], lambda_minus_set[1], lambda_minus_set[2],
        path_mask, L
    )

    stack_path_result = 255 - raw_result

    confirmed_mask = compute_confirmed_mask(stack_path_result)

    refined_mask = refine_mask(
        confirmed_mask,
        closing_kernel_size=3
    )

    soft_mask = build_soft_mask_from_levels(
        raw_result=raw_result,
        allowed_mask=refined_mask,
        levels_per_mask=levels_per_mask
    )

    result = enhance_with_soft_mask(
        original_img=img,
        soft_mask=soft_mask,
        darken_value=40
    )

    return result, processed_img, stack_path_result, confirmed_mask, refined_mask, soft_mask


def compute_confirmed_mask(result_img):
    confirmed = cv2.adaptiveThreshold(
        result_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    return (confirmed > 0).astype(np.uint8)


def refine_mask(confirmed_mask, closing_kernel_size=3):
    if closing_kernel_size % 2 == 0:
        closing_kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_kernel_size, closing_kernel_size)
    )

    refined_mask = cv2.morphologyEx(
        confirmed_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel
    )

    return refined_mask


def build_soft_mask_from_levels(raw_result, allowed_mask, levels_per_mask=30):
    soft_mask = np.zeros_like(raw_result, dtype=np.float32)

    max_level = int(np.max(raw_result))

    if max_level == 0:
        return soft_mask

    allowed_mask = allowed_mask.astype(np.float32)

    for start in range(1, max_level + 1, levels_per_mask):
        end = min(start + levels_per_mask, 256)

        level_mask = (
            (raw_result >= start) &
            (raw_result < end)
        )

        weight = end / 255.0

        soft_mask += level_mask.astype(np.float32) * weight * allowed_mask

    return np.clip(soft_mask, 0.0, 1.0)


def enhance_with_soft_mask(original_img, soft_mask, darken_value=40):
    img_float = original_img.astype(np.float32)
    enhanced = img_float - soft_mask * darken_value

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
                pred_levels,
                pred_lambdas,
                pred_sizes,
                pred_count
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
def update_lambda_minus_set(img, mask):
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
                pred_levels,
                pred_lambdas,
                pred_sizes,
                pred_count
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
    img = cv2.imread("./dataset/1.bmp", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Не вдалося завантажити зображення")

    result, processed_img, stack_path_result, confirmed_mask, refined_mask, soft_mask = handwriting_enhancement_algorithm(
        img,
        L=10,
        levels_per_mask=30
    )

    plt.figure(figsize=(22, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Оригінал")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(processed_img, cmap="gray")
    plt.title("Median blur")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(stack_path_result, cmap="gray")
    plt.title("Stack path result")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(refined_mask * 255, cmap="gray")
    plt.title("Refined mask")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(soft_mask, cmap="gray")
    plt.title("Soft mask")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(result, cmap="gray")
    plt.title("Результат")
    plt.axis("off")

    plt.tight_layout()
    plt.show()