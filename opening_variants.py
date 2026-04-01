from skimage.morphology import area_opening, reconstruction
import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

#opening by reconstruction
img = cv2.imread("./dataset/8.bmp", 0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

img_erosion = cv2.erode(img, kernel, iterations=1)
opening_r = reconstruction(img_erosion, img)

#parametric opening
opening_param = np.zeros_like(img)

coords = np.argwhere(kernel == 1)
subsets = list(combinations(coords, 4))

for subset in subsets:
    sub_kernel = np.zeros_like(kernel, dtype=np.uint8)

    for r, c in subset:
        sub_kernel[r, c] = 1

    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, sub_kernel)
    opening_param = np.maximum(opening_param, opened)

#attribute opening
opening_attribute = np.zeros_like(img)
threshold = 160

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary)

for i in range(1, num_labels):
    mask = (labels == i)
    component_pixels = img[mask]

    if component_pixels.mean() < threshold:
        opening_attribute[mask] = component_pixels


#area opening
opening_area = area_opening(img, area_threshold=100)


#plotting - figure 1

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Оригінальне зображення")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(opening_r, cmap="gray")
plt.title("Розкриття за реконструкцією")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(img, cmap="gray")
plt.title("Оригінальне зображення")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(opening_param, cmap="gray")
plt.title("Параметричне розкриття")
plt.axis("off")

plt.tight_layout()
plt.show()

#figure 2

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Оригінальне зображення")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(opening_attribute, cmap="gray")
plt.title("Розкриття за характеристикою")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(img, cmap="gray")
plt.title("Оригінальне зображення")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(opening_area, cmap="gray")
plt.title("Розкриття за площею")
plt.axis("off")

plt.tight_layout()
plt.show()