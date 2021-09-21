from PIL.Image import new
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

    new_image = np.dstack((new_image_lightness, new_image_average, new_image_luminosity, new_image_opencv))

    return new_image


def rgb2opponent(input_image):
    # converts an RGB image into opponent colour space
    
    [R, G, B] = getColourChannels(input_image)

    O1 = (R - G) / np.sqrt(2)
    O2 = (R + G - 2 * B) / np.sqrt(6)
    O3 = (R + G + B) / np.sqrt(3)

   
    # Normalise the values to be in the range of [0-255]

    O1 = (O1 + (255 / np.sqrt(2))) * ((np.sqrt(2) * 255) / 510)
    O2 = (O2 + (510 / np.sqrt(6))) * ((np.sqrt(6) * 255) / 1020)
    O3 = O3 * np.sqrt(3) / 3
    

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
    new_image = np.nan_to_num(new_image)
    new_image = new_image * 255
    return new_image
