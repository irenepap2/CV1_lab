import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from scipy import signal 

# 1. Divide input images on non-overlapping regions, each region being 15x15.
# 2. For each region compute A, AT and b. Then, estimate optical flow as given in Equation 22.
# 3. When you have estimation for optical flow (Vx; Vy) of each region, you
# should display the results. There is a matplotlib function quiver which
# plots a set of two-dimensional vectors as arrows on the screen. Try to
# figure out how to use this to show your optical flow results.

def calculate_derivatives(I_t0, I_t1):
    # Obtain image x,y-derivatives.
    first_derivative = np.array([[-1, 0, 1]])

    I_x = signal.convolve2d(I_t1, first_derivative, mode='same')
    I_y = signal.convolve2d(I_t1, first_derivative.T, mode='same')

    # Obtain time derivative as the subtraction between frames.
    I_t = I_t0 - I_t1

    return I_x, I_y, I_t

def load_images(name_image_t0, name_image_t1):
    image_dir = './images/'

    # Load the two images
    I_t0 = cv2.imread(image_dir + name_image_t0,0)
    I_t1 = cv2.imread(image_dir + name_image_t1,0)
    original_image = cv2.imread(image_dir + name_image_t1)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Convert the to np.float32
    I_t0 = I_t0.astype(np.float32)
    I_t1 = I_t1.astype(np.float32)

    return original_image, I_t0, I_t1

def calculate_optical_flow(I_x, I_y, I_t):

    V_x = []
    V_y = []

    # Calculate Vx, Vy for each
    for i in range(len(I_x)):
        sub_I_x = I_x[i].flatten()
        sub_I_y = I_y[i].flatten()

        A = np.array((sub_I_x, sub_I_y)).T
        b = -I_t[i].flatten()
        tempVx, tempVy = np.linalg.inv(A.T@A)@A.T@b
        V_x.append(tempVx)
        V_y.append(tempVy)

    return V_x, V_y


if __name__ == '__main__':

    region_size = 15

    original_image, I_t0, I_t1 = load_images('Car1.jpg', 'Car2.jpg')

    I_x, I_y, I_t = calculate_derivatives(I_t0, I_t1)

    h,w = I_t0.shape

    horizontal_subregions = h // region_size
    vertical_subregions = w // region_size
    number_of_subregions = horizontal_subregions * vertical_subregions
    # subregions = np.zeros(horizontal_subregions, vertical_subregions, horizontal_subregions * vertical_subregions)
    sub_I_x = []
    sub_I_y = []
    sub_I_t = []
    
    for i in range(horizontal_subregions):
        h_begin = i*region_size
        h_end   = (i+1)*region_size
        for j in range(vertical_subregions):
            v_begin = j*region_size
            v_end   = (j+1)*region_size
            sub_I_x.append(I_x[h_begin : h_end, v_begin : v_end])
            sub_I_y.append(I_y[h_begin : h_end, v_begin : v_end])
            sub_I_t.append(I_t[h_begin : h_end, v_begin : v_end])
  

    V_x, V_y = calculate_optical_flow(sub_I_x, sub_I_y, sub_I_t)
    
    subregion_indices = []
    for i in range(horizontal_subregions):
        for j in range(vertical_subregions):
            y = i*region_size  + region_size // 2
            x = j*region_size  + region_size // 2
            subregion_indices.append((x, y))

    subregion_indices = np.array(subregion_indices)    
    
    
    plt.figure()
    plt.imshow(original_image)
    plt.quiver(subregion_indices[:,0], subregion_indices[:,1], V_x, V_y, angles='xy', scale_units='xy', scale=0.1)
    plt.show()



    









