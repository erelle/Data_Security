import math
from PIL import Image as im

import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt

print("DST: ")
img = cv2.imread('strawberries.jpeg', 0)
img1 = img.astype('float')

C_temp = np.zeros(img.shape)
dst = np.zeros(img.shape)

m, n = img.shape
N = n
for i in range(m):
    for j in range(n):
        C_temp[i, j] = np.sin(np.pi * (i + 1) * (j + 1) / (N + 1)) * np.sqrt(2 / (N + 1))

dst = np.dot(C_temp, img1)
dst = np.dot(dst, np.transpose(C_temp))
dst1 = np.log(abs(dst))  # do log processing
img_recor = np.dot(np.transpose(C_temp), dst)
img_recor = np.dot(img_recor, C_temp)

Y = np.square(np.subtract(img1,img_recor)).mean()
print("MSE value of DST method:", Y)

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

print("PSNR value of DST method: ",PSNR(img1, img_recor))

plt.imshow(img_recor, 'gray',aspect='auto')
plt.savefig("dst.jpeg", dpi=300, bbox_inches='tight')
