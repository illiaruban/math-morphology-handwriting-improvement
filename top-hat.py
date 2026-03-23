#in this file white and black top-hat operations are applied to image
import cv2
import matplotlib.pyplot as plt
import numpy as np

kernel = np.ones((5, 5), np.uint8)

#white top-hat
img1 = cv2.imread('./dataset/2.bmp', 0)
img1 = cv2.bitwise_not(img1)

tophat_img = cv2.morphologyEx(img1, cv2.MORPH_TOPHAT, kernel)

#black top-hat
img2 = cv2.imread('./dataset/3.bmp', 0)

black_tophat_img = cv2.morphologyEx(img2, cv2.MORPH_BLACKHAT, kernel)

#plotting
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title("Оригінальне зображення 1")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(tophat_img, cmap='gray')
plt.title("White top-hat")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img2, cmap='gray')
plt.title("Оригінальне зображення 2")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(black_tophat_img, cmap='gray')
plt.title("Black top-hat")
plt.axis('off')

plt.tight_layout()
plt.show()