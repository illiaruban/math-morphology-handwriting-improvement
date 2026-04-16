import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq

def talbot_algorithm(img, L, reduce_gray_step=2, dark_text=True):

    if img.dtype != np.uint8:
        raise ValueError("img must be a uint8 grayscale image")

    # invert if we want to preserve dark handwritten text
    work = 255 - img if dark_text else img.copy()

    # optional gray-level reduction
    if reduce_gray_step > 1:
        work = (work // reduce_gray_step) * reduce_gray_step

    # sort all gray levels in increasing order
    levels = np.unique(work)
    levels.sort()

    # initialize lambda_plus/lambda_minus to max path length
    height, width = work.shape
    max_len = height + width

    lambda_plus = np.full((height, width), max_len, dtype=np.int32)
    lambda_minus = np.full((height, width), max_len, dtype=np.int32)

    # opening transform in working-image domain
    opening_work = np.zeros((height, width), dtype=np.uint8)

    # initial survive mask
    current_total = lambda_plus + lambda_minus - 1
    current_survive = (work >= levels[0]) & (current_total >= L)

    # apply update algorithms for each gray level and record changes
    for level in levels[:-1]:
        lambda_plus = update_lambda_plus(work, level, lambda_plus)
        lambda_minus = update_lambda_minus(work, level, lambda_minus)

        next_total = lambda_plus + lambda_minus - 1
        next_survive = (work > level) & (next_total >= L)

        lost = current_survive & (~next_survive)
        opening_work[lost] = level

        current_survive = next_survive

    # pixels that still survive at the last level
    opening_work[current_survive] = levels[-1]

    # convert back to original contrast if needed
    result = 255 - opening_work if dark_text else opening_work

    return result

def update_lambda_plus(img, level, lambda_plus):
    height, width = img.shape

    # pixels that stay in the next upper level set
    alive_next = img > level

    # priority queue with all pixels at the current gray level
    Q = []
    current_pixels = np.argwhere(img == level)

    for x, y in current_pixels:
        # process from bottom to top for lambda_plus
        heapq.heappush(Q, ((-x, y), x, y))

    while Q:
        _, x, y = heapq.heappop(Q)

        # lambda = 0
        lambda_val = 0

        # if f(x) is above current grey level
        if img[x, y] > level:
            max_pred = 0

            nx = x + 1
            if nx < height:
                # bottom-left
                ny = y - 1
                if ny >= 0 and alive_next[nx, ny]:
                    val = lambda_plus[nx, ny]
                    if val > max_pred:
                        max_pred = val

                # bottom
                ny = y
                if alive_next[nx, ny]:
                    val = lambda_plus[nx, ny]
                    if val > max_pred:
                        max_pred = val

                # bottom-right
                ny = y + 1
                if ny < width and alive_next[nx, ny]:
                    val = lambda_plus[nx, ny]
                    if val > max_pred:
                        max_pred = val

            lambda_val = max_pred + 1

        # if lambda < lambda_plus(x)
        if lambda_val < lambda_plus[x, y]:
            # step 8
            lambda_plus[x, y] = lambda_val

            # push successors (top-left, top, top-right)
            nx = x - 1
            if nx >= 0:
                ny = y - 1
                if ny >= 0 and alive_next[nx, ny]:
                    heapq.heappush(Q, ((-nx, ny), nx, ny))

                ny = y
                if alive_next[nx, ny]:
                    heapq.heappush(Q, ((-nx, ny), nx, ny))

                ny = y + 1
                if ny < width and alive_next[nx, ny]:
                    heapq.heappush(Q, ((-nx, ny), nx, ny))

    return lambda_plus


def update_lambda_minus(img, level, lambda_minus):
    height, width = img.shape

    # pixels that stay in the next upper level set
    alive_next = img > level

    # priority queue with all pixels at the current gray level
    Q = []
    current_pixels = np.argwhere(img == level)

    for x, y in current_pixels:
        # process from top to bottom for lambda_minus
        heapq.heappush(Q, ((x, y), x, y))

    while Q:
        _, x, y = heapq.heappop(Q)

        # lambda = 0
        lambda_val = 0

        # if f(x) is above current grey level
        if img[x, y] > level:
            max_pred = 0

            nx = x - 1
            if nx >= 0:
                # top-left
                ny = y - 1
                if ny >= 0 and alive_next[nx, ny]:
                    val = lambda_minus[nx, ny]
                    if val > max_pred:
                        max_pred = val

                # top
                ny = y
                if alive_next[nx, ny]:
                    val = lambda_minus[nx, ny]
                    if val > max_pred:
                        max_pred = val

                # top-right
                ny = y + 1
                if ny < width and alive_next[nx, ny]:
                    val = lambda_minus[nx, ny]
                    if val > max_pred:
                        max_pred = val

            lambda_val = max_pred + 1

        # if lambda < lambda_minus(x)
        if lambda_val < lambda_minus[x, y]:
            lambda_minus[x, y] = lambda_val

            # push successors (bottom-left, bottom, bottom-right)
            nx = x + 1
            if nx < height:
                ny = y - 1
                if ny >= 0 and alive_next[nx, ny]:
                    heapq.heappush(Q, ((nx, ny), nx, ny))

                ny = y
                if alive_next[nx, ny]:
                    heapq.heappush(Q, ((nx, ny), nx, ny))

                ny = y + 1
                if ny < width and alive_next[nx, ny]:
                    heapq.heappush(Q, ((nx, ny), nx, ny))

    return lambda_minus

if __name__ == "__main__":
    img = cv2.imread("./dataset/10.bmp", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Не вдалося завантажити зображення")

    #apply algorithm
    result1 = talbot_algorithm(img, L=10, reduce_gray_step=2, dark_text=True)
    result2 = talbot_algorithm(img, L=20, reduce_gray_step=2, dark_text=True)
    result3 = talbot_algorithm(img, L=30, reduce_gray_step=2, dark_text=True)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Зображення з рівнями сірого")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(result1, cmap="gray")
    plt.title("Алгоритм Талбота (L = 10)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(result2, cmap="gray")
    plt.title("Алгоритм Талбота (L = 20)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(result3, cmap="gray")
    plt.title("Алгоритм Талбота (L = 30)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()