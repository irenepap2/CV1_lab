import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import maximum_filter
from harris_corner_detector import *
from lucas_kanade import calculate_optical_flow_with_LK_for_corners
import os

def save_video(video_name, frames, width, height):
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for frame in frames:
        video.write(frame)

def update_corners(r, c, Vx, Vy):
    r = np.round(r + Vy).astype(np.int)
    c = np.round(c + Vx).astype(np.int)
    return r, c

def tracking(path):
    files = os.listdir(path)
    img = cv2.imread(os.path.join(path, files[0]), 0)
    W, H = img.shape
    # detect corners of first image
    H, r, c = harris_corner_detector(img, 1, 3, 0.001, 5, plot=False)
    figure = plt.figure()
    frames = []
    for i in range(1, len(files)):
        cur_img = cv2.imread(os.path.join(path, files[i]), 0)
        subregion_indices, V_x, V_y, r, c = calculate_optical_flow_with_LK_for_corners(files[i-1], files[i], path, r, c) 
        r, c = update_corners(r, c, V_x, V_y)
        plt.clf()
        plt.imshow(cur_img)
        plt.quiver(subregion_indices[:,0], subregion_indices[:,1], V_x, V_y, angles='xy', scale_units='xy')
        plt.draw()
        plt.pause(0.0001)
        frame = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        frames.append(frame)

    save_video("output.avi", frames, W, H)


if __name__ == '__main__':
    path = './lab3/images/toy/'
    tracking(path)


