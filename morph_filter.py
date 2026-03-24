#in this file three morphological filters are applied to images

import cv2
import matplotlib.pyplot as plt
import numpy as np

#parallel morphological filter
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


#sequential morphological filter
img = cv2.imread("./dataset/5.bmp", 0)

kernel = np.ones((4,4), np.uint8)

#opening
img_erosion = cv2.erode(img, kernel, iterations=1)
img_opening = cv2.dilate(img_erosion, kernel, iterations=1)

#dual closing
inv_img = cv2.bitwise_not(img_opening)
img_erosion_inv = cv2.erode(inv_img, kernel, iterations=1)
img_closing_inv = cv2.dilate(img_erosion_inv, kernel, iterations=1)

result = cv2.bitwise_not(img_closing_inv)

#plotting
plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_opening, cmap='gray')
plt.title("Після розкриття")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(inv_img, cmap='gray')
plt.title("Комплімент результату розкриття")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(result, cmap='gray')
plt.title("Дуальне закриття")
plt.axis('off')

plt.tight_layout()
plt.show()


#iterative morphological filter
img = cv2.imread("./dataset/6.bmp", 0)
current = img.copy()
first_iter = None

for i in range(10):
    k = 1 + 4*i  
    kernel = np.ones((k, k), np.uint8)

    # opening
    img_erosion = cv2.erode(current, kernel, iterations=1)
    img_opening = cv2.dilate(img_erosion, kernel, iterations=1)

    # dual closing
    inv_img = cv2.bitwise_not(img_opening)
    img_erosion_inv = cv2.erode(inv_img, kernel, iterations=1)
    img_closing_inv = cv2.dilate(img_erosion_inv, kernel, iterations=1)
    result = cv2.bitwise_not(img_closing_inv)

    if i == 0:
        first_iter = result.copy()

    if np.array_equal(current, result):
        break

    current = result.copy()

last_iter = current

#plotting
plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(first_iter, cmap='gray')
plt.title("Перша ітерація")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(last_iter, cmap='gray')
plt.title("Остання ітерація")
plt.axis('off')

plt.tight_layout()
plt.show()