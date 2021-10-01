import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import maximum_filter
from gauss2D import *
import scipy.stats as st

def get_harris_points(H):
    H_local_maxima = maximum_filter(H, size=5)
    H[H < H_local_maxima] = 0.0
    r, c = np.where(H > 0.001)
    return r, c

def max_of_neighbors(H, radius, rowNumber, columnNumber):
    neighbors = [[H[i][j] if  i >= 0 and i < len(H) and j >= 0 and j < len(H[0]) else 0
    for j in range(columnNumber-1-radius, columnNumber+radius)]
      for i in range(rowNumber-1-radius, rowNumber+radius)]
    return np.max(np.array(neighbors))

def harris_corner_detector(img, gauss_sigma, gauss_kernel_size, threshold, window):
    
    # convert from uint8 to float32
    img = img.astype(np.float32) / 255
    
    #computing partial derivatives Ix and Iy
    Gx = np.array([[-1, 0, 1]])
    Ix = signal.convolve2d(img, Gx, mode='same')
    Iy = signal.convolve2d(img, Gx.T, mode='same')

    #Gaussian kernel using previous's excercise gauss2D function
    gauss2d = gauss2D(gauss_sigma, gauss_sigma, gauss_kernel_size)
    
    #computing elements of Q matrix (A, B, C)
    A = signal.convolve2d(Ix ** 2, gauss2d, mode='same')
    B = signal.convolve2d(Ix * Iy, gauss2d, mode='same')
    C = signal.convolve2d(Iy ** 2, gauss2d, mode='same')

    #computing H
    H = (A * C - B**2) - 0.04 * (A + C)**2

    #computing local maxima of H
    # r: row indices of corners
    # c: column indices of corners
    r = []
    c = []
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if (H[i][j] == max_of_neighbors(H, window, i, j) and H[i][j] > threshold):
                r.append(i)
                c.append(j)
    
    #r,c  = get_harris_points(H)
    plot_figures(img, Ix, Iy, H, r, c)
    return H, r, c

def plot_figures(img, Ix, Iy, H, r, c):   
    fig, (ix, iy, corners) = plt.subplots(1, 3, figsize=(12, 5))
    ix.imshow(Ix, cmap="gray")
    ix.set_title('Gradient in x-direction')
    ix.set_axis_off()
    iy.imshow(Iy, cmap="gray")
    iy.set_title('Gradient in y-direction')
    iy.set_axis_off()
    corners.imshow(img, cmap="gray")
    corners.scatter(c, r, s=1, color='red')
    corners.set_title('Corners')
    corners.set_axis_off()
    plt.show()


if __name__ == '__main__':
    im = cv2.imread('./images/toy/0001.jpg', 0)
    # detect corners and show results
    H, r, c = harris_corner_detector(im, 1, 3, 0.001, 5)
