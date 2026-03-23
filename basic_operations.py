#in this file the most basic operations of mathematical morphology are applied to image

import cv2
import matplotlib.pyplot as plt
import numpy as np

#download image 1
img = cv2.imread('./dataset/1.bmp', 0)
img = cv2.bitwise_not(img)
kernel = np.ones((5, 5), np.uint8)

#erosion
img_erosion = cv2.erode(img, kernel, iterations=1)

#dilation
img_dilation = cv2.dilate(img, kernel, iterations=1)

#opening
img_opening = cv2.dilate(img_erosion, kernel, iterations=1)

#closing
img_closing = cv2.erode(img_dilation, kernel, iterations=1)

#plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_erosion, cmap='gray')
plt.title("Після звуження")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_dilation, cmap='gray')
plt.title("Після розширення")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_opening, cmap='gray')
plt.title("Після розкриття")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_closing, cmap='gray')
plt.title("Після закриття")
plt.axis('off')

plt.tight_layout()
plt.show()


# --- Erosion ---
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_erosion, cmap='gray')
plt.title("Після звуження")
plt.axis('off')

plt.tight_layout()
plt.show()


# --- Dilation ---
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_dilation, cmap='gray')
plt.title("Після розширення")
plt.axis('off')

plt.tight_layout()
plt.show()


# --- Opening ---
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_opening, cmap='gray')
plt.title("Після розкриття")
plt.axis('off')

plt.tight_layout()
plt.show()


# --- Closing ---
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Оригінальне зображення")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_closing, cmap='gray')
plt.title("Після закриття")
plt.axis('off')

plt.tight_layout()
plt.show()