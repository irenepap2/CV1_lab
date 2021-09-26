import numpy as np
import cv2
import matplotlib.pyplot as plt
from gauss2D import *
from scipy import signal
import scipy.ndimage as nd

def log2D(sigma, kernel_size):
    LoG = np.zeros((kernel_size, kernel_size))
    x = np.arange(-np.floor(kernel_size/2), np.floor(kernel_size/2)+1, 1)
    y = np.arange(-np.floor(kernel_size/2), np.floor(kernel_size/2)+1, 1)

    for x_i in x:
        for y_i in y:
            LoG[np.where(x == x_i), np.where(y == y_i)] = -1/(np.pi * sigma**4) * (1 - (x_i**2 + y_i**2)/(2*sigma**2))*(np.exp(-(x_i**2+y_i**2)/(2*sigma**2)))
    
    print(LoG)
    #normalize
    LoG = LoG/np.sum(LoG)
    return LoG

def compute_LoG(image, LOG_type):

    if LOG_type == 1:
        #method 1
        G = gauss2D(0.5, 0.5, 5)
        G = signal.convolve2d(image, G, mode='same')
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1 , 0]])
        imOut = signal.convolve2d(image, laplacian, mode='same')
        return imOut

    elif LOG_type == 2:
        #method 2
        #LoG = log2D(0.5, 5)
        # we found a 5x5 LoG kernel on the internet, which gives better results
        LoG = np.array([[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2 , 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]])
        imOut = signal.convolve2d(image, LoG, mode='same')
        return imOut

    elif LOG_type == 3:
        #method 3
        G1 = gauss2D(0.5, 0.5, 5)
        #1.6 times σ1
        G2 = gauss2D(0.8, 0.8, 5)
        DoG = G1 - G2
        imOut = signal.convolve2d(image, DoG, mode='same')
        return imOut

if __name__ == '__main__':
    im = cv2.imread('./Image_enhancement/images/image2.jpg', 0)

    imOut = compute_LoG(im, LOG_type=1)

    figure = plt.figure()
    plt.imshow(imOut, cmap="gray")
    plt.axis('off')
    plt.show()
