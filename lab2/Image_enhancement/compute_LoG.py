import numpy as np
import cv2
import matplotlib.pyplot as plt
from gauss2D import *
from scipy import signal

def log2D(sigma, kernel_size):
    LoG = np.zeros((kernel_size, kernel_size))
    x = np.arange(-np.floor(kernel_size/2), np.floor(kernel_size/2)+1, 1)
    y = np.arange(-np.floor(kernel_size/2), np.floor(kernel_size/2)+1, 1)

    for x_i in x:
        for y_i in y:
            LoG[np.where(x == x_i), np.where(y == y_i)] = -1/(np.pi * sigma **4) * (1 - (x_i**2 + y_i**2)/(2*sigma**2))*(np.exp(-(x_i**2+y_i**2)/(2*sigma**2)))
    
    #normalize
    LoG = LoG/np.sum(LoG)
    return LoG

def compute_LoG(image, LOG_type):

    if LOG_type == 1:
        #method 1
        G = gauss2D(0.5, 0.5, 5)
        G = signal.convolve2d(image, G, mode='same')
        imOut = cv2.Laplacian(G, ddepth=-1)
        return imOut

    elif LOG_type == 2:
        #method 2
        LoG = log2D(0.5, 5)
        imOut = signal.convolve2d(image, LoG, mode='same')
        return imOut

    elif LOG_type == 3:
        #method 3
        G1 = gauss2D(0.5, 0.5, 5)
        G2 = gauss2D(1, 1, 5)
        DoG = G1 - G2
        imOut = signal.convolve2d(image, DoG, mode='same')
        return imOut


if __name__ == '__main__':
    im = cv2.imread('./lab2/Image_enhancement/images/image2.jpg', 0)

    imOut = compute_LoG(im, LOG_type=2)

    figure = plt.figure()
    plt.imshow(imOut, cmap="gray")
    plt.axis('off')
    plt.show()
