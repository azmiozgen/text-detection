import cv2
import matplotlib.pyplot as plt
import numpy as np


def apply_canny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)

def plt_show(*images):
    count = len(images)
    nRow = np.ceil(count / 3.)
    for i in range(count):
        plt.subplot(nRow, 3, i + 1)
        if len(images[i][0].shape) == 2:
            plt.imshow(images[i][0], cmap='gray')
        else:
            plt.imshow(images[i][0])
        plt.xticks([])
        plt.yticks([])
        plt.title(images[i][1])
    plt.show()