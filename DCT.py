from scipy.fftpack import dct, idct
import math
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pylab as plt
print("DCT: \n")
def dct2(a):# implement 2D DCT
    return dct(dct(a.T, norm='ortho').T, norm='ortho')
def idct2(a):# implement 2D IDCT
    return idct(idct(a.T, norm='ortho').T, norm='ortho')
# read lena RGB image and convert to grayscale
im = rgb2gray(imread('strawberries.jpeg'))
imF = dct2(im)
im1 = idct2(imF)
print("are the images equal? :",np.allclose(im, im1)) # check if the reconstructed image is nearly equal to the original image-True
Y = np.square(np.subtract(im,im1)).mean() #mse calculation:
print("MSE value of DCT method:", Y)
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal,Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

print("PSNR value of DCT method: ",PSNR(im, im1))
plt.gray()# plot original and reconstructed images with matplotlib.pylab
plt.imshow(im1), plt.axis('off')
plt.savefig("dct.png")
plt.show()

