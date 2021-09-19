# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 19:00:04 2021

@author: silav
"""

import os
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors


def normalize(arr):
    ''' Function to scale an input array to [0, 1] '''
    arr_min = arr.min()
    arr_max = arr.max()
    # Check the original min and max values
    print('Min: %.3f, Max: %.3f' % (arr_min, arr_max))
    arr_range = arr_max - arr_min
    arr_new = np.array((arr-arr_min) / float(arr_range), dtype='f')
    # Make sure min value is 0 and max value is 1
    print('Min: %.3f, Max: %.3f' % (arr_new.min(), arr_new.max()))
    return arr_new

def main():
    colours = []
    #Read the inputs
    original = cv2.imread("C:/Users/silav/Downloads/lab1/intrinsic_images/ball.png")
    albedo = cv2.imread("C:/Users/silav/Downloads/lab1/intrinsic_images/ball_albedo.png")
    shading = cv2.imread("C:/Users/silav/Downloads/lab1/intrinsic_images/ball_shading.png")
    #BGR to RGB
    original_rgb=cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    albedo_rgb=cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
    shading_rgb=cv2.cvtColor(shading, cv2.COLOR_BGR2RGB)
    #split colors to RGB space
    r, g, b = cv2.split(original_rgb)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = original_rgb.reshape((np.shape(original_rgb)[0]*np.shape(original_rgb)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()
    #Uniform true matterial colors
    red = max(np.unique(albedo[:,:,0]))
    green = max(np.unique(albedo[:,:,1]))
    blue = max(np.unique(albedo[:,:,2]))
    #Transform all blue and red pixels to green
    green_albedo = np.where(albedo_rgb==108, 0, albedo_rgb)
    green_albedo = np.where(green_albedo==184, 0, green_albedo)
    green_albedo = np.where(green_albedo==141, 255, green_albedo)
    #Normalize to [0, 1] to prevent overflow
    green_albedo_norm = np.array(green_albedo)
    green_albedo_norm = normalize(green_albedo_norm)
    shading_rgb_norm = np.array(shading_rgb)
    shading_rgb_norm = normalize(shading_rgb_norm)

    #Element wise multiplication
    reconst_green_norm = np.multiply(green_albedo_norm, shading_rgb_norm)
    reconst_green = cv2.normalize(reconst_green_norm, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    reconst_green = reconst_green.astype(np.uint8)
    
    #plotting images
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    plt.subplot(1, 2, 1) 
    plt.imshow(original_rgb)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 2, 2) 
    plt.imshow(reconst_green)
    plt.title('Reconstruction')
    plt.axis('off')
    plt.show()
    
    
if __name__ == "__main__":
    main()