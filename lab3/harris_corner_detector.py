import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import maximum_filter
from gauss2D import *
from lucas_kanade import *

def get_corner_points(H, threshold):
    # using scipy.ndimage.filters.maximum_filter to compute the local maxima of H
    H_local_maxima = maximum_filter(H, size=5)
    # we set the values that are less than out threshold to -1
    H[H < threshold] = -1
    # we take the rows and columns where our H - H_local_maxima == 0, so when our centered point is the maximum
    r, c = np.where(H - H_local_maxima == 0) 
    return r, c

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
    
    #compute corner points
    r,c  = get_corner_points(H, threshold)

    plot_figures(img, Ix, Iy, H, r, c)
    return H, r, c

def plot_figures(img, Ix, Iy, H, r, c):   
    fig, (ix, iy, corners) = plt.subplots(1, 3, figsize=(12, 5))
    ix.imshow(Ix)
    ix.set_title('Gradient in x-direction')
    ix.set_axis_off()
    iy.imshow(Iy)
    iy.set_title('Gradient in y-direction')
    iy.set_axis_off()
    corners.imshow(img)
    corners.scatter(c, r, s=1, color='red')
    corners.set_title('Corners')
    corners.set_axis_off()
    plt.show()


if __name__ == '__main__':
    im = cv2.imread('./images/toy/0001.jpg', 0)
    # detect corners and show results
    H, r, c = harris_corner_detector(im, 1, 3, 0.001, 5)