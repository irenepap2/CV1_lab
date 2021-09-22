import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

def compute_LoG(image, LOG_type):

    if LOG_type == 1:
        #method 1
        print('Not implemented\n')

    elif LOG_type == 2:
        #method 2
        print('Not implemented\n')

    elif LOG_type == 3:
        #method 3
        print('Not implemented\n')

	return imOut

if __name__ == '__main__':
    im = cv2.imread('./lab2/Image_enhancement/images/image2.jpg', 0)
    imOut = compute_LoG(im)