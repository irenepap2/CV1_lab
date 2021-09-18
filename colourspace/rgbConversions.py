from getColourChannels import getColourChannels
import numpy as np
import cv2
import matplotlib.pyplot as plt

def rgb2grays(input_image):
    # converts an RGB into grayscale by using 4 different methods
    [R, G, B] = getColourChannels(input_image)
    # ligtness method
    new_image_lightness = (np.max(np.stack((R, G, B)), axis=0) + np.min(np.stack((R, G, B)), axis=0)) / 2.
    # average method
    new_image_average = (R + G + B) / 3
    # luminosity method
    new_image_luminosity = (0.21 * R) + (0.72 * G) + (0.07 * B)
    # built-in opencv function 
    new_image_opencv = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    figure = plt.figure()
    ax1 = figure.add_subplot(141)
    ax1.imshow(new_image_lightness,cmap='gray', vmin=0, vmax=255)
    ax2 = figure.add_subplot(142)
    ax2.imshow(new_image_average,cmap='gray', vmin=0, vmax=255)
    ax3 = figure.add_subplot(143)
    ax3.imshow(new_image_luminosity,cmap='gray', vmin=0, vmax=255)
    ax4 = figure.add_subplot(144)
    ax4.imshow(new_image_opencv,cmap='gray', vmin=0, vmax=255)
    plt.show(block=False)

    return new_image_opencv


def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space
    
    [R, G, B] = getColourChannels(input_image)

    O1 = (R - G) / np.sqrt(2)
    O2 = (R + G - 2 * B) / np.sqrt(6)
    O3 = (R + G + B) / np.sqrt(3)

    new_image = np.dstack((O1, O2, O3))

    return new_image


def rgb2normedrgb(input_image):
    # converts an RGB image into normalized rgb colour space

    [R, G, B] = getColourChannels(input_image)
    total_rgb = R + G + B
    r = R / total_rgb
    g = G / total_rgb
    b = B / total_rgb

    new_image = np.dstack((r, g, b))

    return new_image
