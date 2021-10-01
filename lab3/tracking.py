import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import maximum_filter
from harris_corner_detector import *
from lucas_kanade import calculate_optical_flow_with_LK_for_corners
import os

def update_corners(r, c, Vx, Vy):
    r += Vy.astype(np.int)
    c += Vy.astype(np.int)
    return r, c

def tracking():
    
    return None

if __name__ == '__main__':
    path = './images/toy/'
    files = os.listdir(path)
    img = cv2.imread(os.path.join(path, files[0]), 0)
    # detect corners and show results
    H, r, c = harris_corner_detector(img, 1, 3, 0.001, 5)
    plt.figure()
    for i in range(1, len(files)):
        cur_img = cv2.imread(os.path.join(path, files[i]), 0)
        subregion_indices, V_x, V_y = calculate_optical_flow_with_LK_for_corners(files[i-1], files[i], path, r, c)
 
        #r, c = update_corners(r, c, V_x, V_y)
        plt.clf()
        plt.imshow(cur_img)
        plt.quiver(subregion_indices[:,0], subregion_indices[:,1], V_x, V_y, angles='xy', scale_units='xy')
        plt.draw()
        plt.pause(0.0001)