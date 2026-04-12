import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


def orientations2D(size=3):
    xx, yy = np.meshgrid(
        range(-(size // 2), size // 2 + 1),
        range(-(size // 2), size // 2 + 1),
        indexing='ij'
    )

    kernel_11 = np.logical_and(xx >= 0, yy >= 0).astype(np.float32)
    kernel_12 = np.logical_and(-xx >= 0, -yy >= 0).astype(np.float32)
    kernel_21 = np.logical_and(-xx >= 0, yy >= 0).astype(np.float32)
    kernel_22 = np.logical_and(xx >= 0, -yy >= 0).astype(np.float32)
    kernel_31 = (xx == 1).astype(np.float32)
    kernel_32 = (-xx == 1).astype(np.float32)
    kernel_41 = (yy == 1).astype(np.float32)
    kernel_42 = (-yy == 1).astype(np.float32)

    kernel_all = np.stack([
        kernel_11, kernel_12,
        kernel_21, kernel_22,
        kernel_31, kernel_32,
        kernel_41, kernel_42
    ])

    kernel_all[:, size // 2, size // 2] = 1
    return kernel_all


class Convo(nn.Module):
    def __init__(self, kernel, input_channels, stride=1, padding=1, dilation=1, requires_grad=False):

        super(Convo, self).__init__()

        self.convfun = F.conv2d
        kernel = kernel.unsqueeze(1).float()

        self.kernel_tensor = th.nn.Parameter(kernel)
        self.kernel_tensor.requires_grad = requires_grad

        self.input_channels = input_channels
        self.dilation = dilation
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.convfun(
            x,
            self.kernel_tensor,
            groups=int(self.input_channels),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )
        return x


def dilation2d(X, conn=8):
    assert conn in [4, 8], "2d connectivity only 4 or 8"

    if conn == 8:
        return F.max_pool2d(X, 3, 1, 1)

    if conn == 4:
        p1 = F.max_pool2d(X, (3, 1), (1, 1), (1, 0))
        p2 = F.max_pool2d(X, (1, 3), (1, 1), (0, 1))
        return th.max(p1, p2)


def erosion2d(X, conn=8):
    return -dilation2d(-X, conn=conn)


def opening2d(X, conn_e=4, conn_d=8):
    return dilation2d(erosion2d(X, conn=conn_e), conn=conn_d)


def closing2d(X, conn_e=8, conn_d=4):
    return erosion2d(dilation2d(X, conn=conn_d), conn=conn_e)


def path_opening2D(I, n_iter, size=3, robustify=False):
    kernel = orientations2D(size=size)
    tkernel = th.from_numpy(kernel)

    gc = Convo(tkernel, 1, padding=size // 2)
    gc.to(I.device)

    oriI = F.relu((gc(I)).clamp_(0, 2) * I - I)
    accI = I.expand_as(oriI) + oriI

    gc = Convo(tkernel, tkernel.shape[0], padding=size // 2)
    gc.to(I.device)

    for _ in range(n_iter):
        oriI = F.relu(gc(oriI).clamp_(0, 2) * I - I)
        if robustify:
            oriI = closing2d(oriI)
        accI += oriI

    b, c, h, w = accI.shape
    oriIsum = th.zeros([b, c // 2, h, w], device=I.device)

    for i in range(0, c - 1, 2):
        oriIsum[:, i // 2] = F.relu(accI[:, i] + accI[:, i + 1] - 1.0)

    openI, _ = oriIsum.max(axis=1, keepdims=True)
    return openI, oriIsum


def rorpo2D(I, n_iter, size=3):
    openI, oriIsum = path_opening2D(I, n_iter, size=size)
    rorpo = th.max(oriIsum, axis=1, keepdims=True)[0] - th.median(oriIsum, axis=1, keepdims=True)[0]
    return rorpo


def minmaxnorm(I):
    old_shape = I.shape
    view_I = I.view(*I.shape[:2], np.prod(I.shape[2:]))

    min_v = view_I.min(axis=-1, keepdims=True)[0]
    max_v = view_I.max(axis=-1, keepdims=True)[0]

    mmI = (view_I - min_v) / (max_v - min_v + 1e-8)
    mmI = mmI.reshape(*old_shape)
    return mmI

if __name__ == "__main__":
    img = cv2.imread("./dataset/7.bmp", cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError("Не вдалося завантажити зображення")

    binary = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 
            31,
            10
        )

    binary = binary.astype(np.float32) / 255.0
    I = th.from_numpy(binary).unsqueeze(0).unsqueeze(0)
    openI, ori = path_opening2D(I, n_iter=5, size=3, robustify=True)

    result = openI.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(12,4))

    plt.subplot(2,2,1)
    plt.title("Оригінальне зображення")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.title("Шляхове розкриття")
    plt.imshow(result, cmap='gray')
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.title("Бінарне зображення")
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.title("Шляхове розкриття")
    plt.imshow(result, cmap='gray')
    plt.axis('off')

    plt.show()