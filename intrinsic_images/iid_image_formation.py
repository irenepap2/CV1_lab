# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 13:09:14 2021

@author: silav
"""
import os
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


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
    #Read the inputs
    original = cv2.imread("C:/Users/silav/Downloads/lab1/intrinsic_images/ball.png")
    albedo = cv2.imread("C:/Users/silav/Downloads/lab1/intrinsic_images/ball_albedo.png")
    shading = cv2.imread("C:/Users/silav/Downloads/lab1/intrinsic_images/ball_shading.png")
    #BGR to RGB
    original_rgb=cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    albedo_rgb=cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
    shading_rgb=cv2.cvtColor(shading, cv2.COLOR_BGR2RGB)
    #Normalize to [0, 1] to prevent overflow
    albedo_rgb_norm = np.array(albedo_rgb)
    albedo_rgb_norm = normalize(albedo_rgb_norm)
    shading_rgb_norm = np.array(shading_rgb)
    shading_rgb_norm = normalize(shading_rgb_norm)

    #Element wise multiplication
    reconst_rgb_norm = np.multiply(albedo_rgb_norm, shading_rgb_norm)
    reconst_rgb = cv2.normalize(reconst_rgb_norm, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    reconst_rgb = reconst_rgb.astype(np.uint8)
    
    #plotting istricts
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    plt.subplot(1, 2, 1) 
    plt.imshow(albedo_rgb)
    plt.title('Albedo')
    plt.axis('off')
    plt.subplot(1, 2, 2) 
    plt.imshow(shading_rgb)
    plt.title('Shading')
    plt.axis('off')
    plt.show()
    #plotting images
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    plt.subplot(1, 2, 1) 
    plt.imshow(original_rgb)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 2, 2) 
    plt.imshow(reconst_rgb)
    plt.title('Reconstruction')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()

