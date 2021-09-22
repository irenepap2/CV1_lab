import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

def compute_gradient(image):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0 , -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = signal.convolve2d(image, sobel_x, boundary='symm', mode='same')
    Gy = signal.convolve2d(image, sobel_y, boundary='symm', mode='same')
    im_magnitude = np.sqrt(np.add(Gx**2, Gy**2))
    im_direction = np.arctan(np.divide(Gy,Gx))
    return Gx, Gy, im_magnitude,im_direction

if __name__ == '__main__':
    im = cv2.imread('./lab2/Image_enhancement/images/image2.jpg', 0)
    Gx, Gy, im_magnitude, im_direction = compute_gradient(im) 

    fig, ([gx, gy], [mag, direc]) = plt.subplots(2, 2, figsize=(6, 15))   
    gx.imshow(Gx, cmap='gray')
    gx.set_title('Gradient in x-direction')
    gx.set_axis_off()
    gy.imshow(Gy, cmap='gray')
    gy.set_title('Gradient in y-direction')
    gy.set_axis_off()
    mag.imshow(im_magnitude, cmap='gray')
    mag.set_title('Gradient magnitude')
    mag.set_axis_off()
    direc.imshow(im_direction, cmap='gray')
    direc.set_title('Gradient direction')
    direc.set_axis_off()
    plt.show()

