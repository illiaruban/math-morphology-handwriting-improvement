import cv2
import numpy as np
import matplotlib.pyplot as plt

def binary_algorithm(binary_img, L):

    result = np.full(binary_img.shape, 255, dtype=np.uint8)
    
    #calculate the max length of path which ends in elements of the image - represented as image
    lambda_plus = compute_lambda_plus(binary_img)
    #calculate the max length of path which starts in elements of the image
    lambda_minus = compute_lambda_minus(binary_img)
    #calculate the max path that includes elements of the image
    lambda_total = lambda_plus + lambda_minus - 1
    #filter the elements of the image that satisfy the condition
    result[(binary_img == 1) & (lambda_total >= L)] = 0

    return result

def get_predecessors(i, j, h, w):
    preds = []

    possible = [
        (i + 1, j - 1),  # bottom-left
        (i + 1, j),      # bottom
        (i + 1, j + 1)   # bottom-right
    ]

    for ni, nj in possible:
        if 0 <= ni < h and 0 <= nj < w:
            preds.append((ni, nj))

    return preds

def compute_lambda_plus(img):
    h, w = img.shape
    lambda_plus = np.zeros((h, w), dtype=np.int32)

    for i in range(h - 1, -1, -1):  # from bottom to top
        for j in range(w):
            if img[i, j] == 0:
                lambda_plus[i, j] = 0
                continue

            preds = get_predecessors(i, j, h, w)

            values = []
            for pi, pj in preds:
                if img[pi, pj] == 1:
                    values.append(lambda_plus[pi, pj])

            if values:
                lambda_plus[i, j] = max(values) + 1
            else:
                lambda_plus[i, j] = 1

    return lambda_plus

def get_successors(i, j, h, w):
    succs = []

    possible = [
        (i - 1, j - 1),  # top-left
        (i - 1, j),      # top
        (i - 1, j + 1)   # top-right
    ]

    for ni, nj in possible:
        if 0 <= ni < h and 0 <= nj < w:
            succs.append((ni, nj))

    return succs

def compute_lambda_minus(img):
    h, w = img.shape
    lambda_minus = np.zeros((h, w), dtype=np.int32)

    for i in range(h): # from top to bottom
        for j in range(w):

            if img[i, j] == 0:
                lambda_minus[i, j] = 0
                continue

            succs = get_successors(i, j, h, w)
            values = []
            for si, sj in succs:
                if img[si, sj] == 1:
                    values.append(lambda_minus[si, sj])

            if values:
                lambda_minus[i, j] = max(values) + 1
            else:
                lambda_minus[i, j] = 1

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

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Оригінальне зображення")

    axes[0, 1].imshow(result1, cmap='gray')
    axes[0, 1].set_title("Результат(L = 10)")

    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title("Бінарне зображення")

    axes[1, 1].imshow(result2, cmap='gray')
    axes[1, 1].set_title("Результат(L = 20)")

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()
