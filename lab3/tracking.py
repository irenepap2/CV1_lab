import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import maximum_filter
from harris_corner_detector import *
from lucas_kanade import calculate_optical_flow_with_LK_for_corners
import os

def frames_to_video(video_name, frames):
    f, height, width, c = frames.shape
    fps = 5
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width,height))
    
    for frame in frames:
        video.write(frame)

def update_corners(r, c, Vx, Vy):
    r = np.round(r + Vy).astype(np.int32)
    c = np.round(c + Vx).astype(np.int32)
    return r, c

def tracking(path, save_video = False):
    files = os.listdir(path)
    img = cv2.imread(os.path.join(path, files[0]), 0)
    height, width = img.shape
    # detect corners of first image
    H, r, c = harris_corner_detector(img, 1, 3, 0.001, 5, plot=False)
    figure = plt.figure()
    # plot initial corners (points of interest)
    plt.clf()
    plt.imshow(img, cmap='gray')
    plt.scatter(c, r, s=1, color='red')
    plt.axis("off")
    plt.pause(0.0001)
    frame = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    frame = np.reshape(frame, figure.canvas.get_width_height()[::-1] + (3,))
    # create list with frames for final video
    frames = [frame]
    for i in range(1, len(files)):
        cur_img = cv2.imread(os.path.join(path, files[i]), 0)
        # compute optical flow for initial corners (points of interest)
        subregion_indices, V_x, V_y, r, c = calculate_optical_flow_with_LK_for_corners(files[i-1], files[i], path, r, c) 
        # update corner points based on V_x and V_y
        r, c = update_corners(r, c, V_x, V_y)
        plt.clf()
        plt.imshow(cur_img, cmap='gray')
        plt.quiver(subregion_indices[:,0], subregion_indices[:,1], V_x, V_y, angles='xy', scale_units='xy', color='red')
        plt.draw()
        plt.axis("off")
        plt.pause(0.0001)
        frame = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        frame = np.reshape(frame, figure.canvas.get_width_height()[::-1]+(3,))
        frames.append(frame)

    frames = np.stack(frames)
    if (save_video):
        frames_to_video("doll.avi", frames)

if __name__ == '__main__':
    path = './images/doll/'
    tracking(path, save_video = True)


