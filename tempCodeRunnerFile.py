import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

MAX_STACK_SIZE = 256


def handwriting_enhancement_algorithm(img, L, levels_per_mask=40):
    if img.dtype != np.uint8:
        raise ValueError("img must be a uint8 grayscale image")

    work = 255 - img
    path_mask = (work > 0).astype(np.uint8)

    lambda_plus_set = update_lambda_plus_set(work, path_mask)
    lambda_minus_set = update_lambda_minus_set(work, path_mask)

    raw_result = build_result(
        lambda_plus_set[0], lambda_plus_set[1], lambda_plus_set[2],
        lambda_minus_set[0], lambda_minus_set[1], lambda_minus_set[2],
        path_mask, L
    )

    level_masks = compute_level_masks(
        raw_result,
        levels_per_mask=levels_per_mask
    )

    result = enhance_with_level_masks(
    original_img=img,
    level_masks=level_masks,
    base_darken_value=5,
    max_darken_value=25
    )

    return result


def compute_level_masks(raw_result, levels_per_mask=40):
    masks = []

    max_level = int(np.max(raw_result))

    for start in range(1, max_level + 1, levels_per_mask):
        end = start + levels_per_mask

        mask = (
            (raw_result >= start) &
            (raw_result < end)
        ).astype(np.uint8)

        if np.any(mask):
            masks.append({
                "range": (start, end),
                "mask": mask
            })

    return sorted(masks, key=lambda x: x["range"][0], reverse=True)


def enhance_with_level_masks(
    original_img,
    level_masks,
    base_darken_value=5,
    max_darken_value=25
):
    enhanced = original_img.astype(np.float32).copy()

    for level_data in level_masks:
        start, end = level_data["range"]
        mask = level_data["mask"].astype(np.uint8)

        strength = end / 255.0

        darken_value = (
            base_darken_value
            + strength * (max_darken_value - base_darken_value)
        )

        mask_pixels = mask == 1

        enhanced[mask_pixels] = np.maximum(
            0,
            enhanced[mask_pixels] - darken_value
        )

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
    img = cv2.imread("./dataset/4.bmp", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Не вдалося завантажити зображення")

    result = handwriting_enhancement_algorithm(
    img,
    L=10,
    levels_per_mask=40
    )

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Оригінальне зображення")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap="gray")
    plt.title("Результат")
    plt.axis("off")

    plt.tight_layout()
    plt.show()