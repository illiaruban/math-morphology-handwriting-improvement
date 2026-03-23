#in this file one of many morphological filters is applied to an image

import cv2
import matplotlib.pyplot as plt
import numpy as np

#download image
img = cv2.imread('./dataset/4.bmp', 0)
img = cv2.bitwise_not(img)
kernel1 = np.ones((5, 5), np.uint8)
kernel2 = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

#opening 1
image_erosion1 = cv2.erode(img, kernel1, iterations=1)
image_opening1 = cv2.dilate(image_erosion1, kernel1, iterations=1)

#opening 2
image_erosion2 = cv2.erode(img, kernel2, iterations=1)
image_opening2 = cv2.dilate(image_erosion2, kernel2, iterations=1)

#union
filtered_img = cv2.bitwise_or(image_opening1, image_opening2)

# plotting
plt.figure(figsize=(10, 12))

plt.subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(image_opening1, cmap='gray')
plt.title("Після розкриття (структурний елемент 1)")
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(img, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(image_opening2, cmap='gray')
plt.title("Після розкриття (структурний елемент 2)")
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(img, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(filtered_img, cmap='gray')
plt.title("Фільтр (об'єднання двох результатів)")
plt.axis('off')

plt.tight_layout()
plt.show()