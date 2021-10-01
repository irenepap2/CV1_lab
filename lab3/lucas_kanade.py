import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from scipy import signal 

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
   

    # Convert the to np.float32
    I_t0 = I_t0.astype(np.float32)
    I_t1 = I_t1.astype(np.float32)

    return  I_t0, I_t1

def calculate_subregions(I_t0, I_x, I_y, I_t, region_size):

    h, w = I_x.shape
    horizontal_subregions = h // region_size
    vertical_subregions = w // region_size
    
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

    return sub_I_x, sub_I_y, sub_I_t

def calculate_subregions(I_t0, I_x, I_y, I_t, region_size):

    h, w = I_t0.shape
    horizontal_subregions = h // region_size
    vertical_subregions = w // region_size
    
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

    return sub_I_x, sub_I_y, sub_I_t

def calculate_subregions_for_corners(I_x, I_y, I_t, r, c, region_size):


    number_of_points = r.shape[0]


    sub_I_x = []
    sub_I_y = []
    sub_I_t = []
    for i in range(number_of_points):
        h_begin = r[i] - (region_size//2)
        h_end   = r[i] + (region_size//2)
        v_begin = c[i] - (region_size//2)
        v_end   = c[i] + (region_size//2)
        sub_I_x.append(I_x[h_begin : h_end, v_begin : v_end])
        sub_I_y.append(I_y[h_begin : h_end, v_begin : v_end])
        sub_I_t.append(I_t[h_begin : h_end, v_begin : v_end])
    

    return sub_I_x, sub_I_y, sub_I_t

def calculate_flow_vectors(I_x, I_y, I_t):

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

def calculate_optical_flow_with_LK(name_image1='Car1.jpg', name_image2='Car2.jpg', region_size=15):
    
    I_t0, I_t1 = load_images(name_image1, name_image2)
    
    I_x, I_y, I_t = calculate_derivatives(I_t0, I_t1)

    sub_I_x, sub_I_y, sub_I_t = calculate_subregions(I_t0, I_x, I_y, I_t, region_size)

    V_x, V_y = calculate_flow_vectors(sub_I_x, sub_I_y, sub_I_t)
    
    h, w = I_x.shape
    horizontal_subregions = h // region_size
    vertical_subregions = w // region_size

    subregion_indices = []
    for i in range(horizontal_subregions):
        for j in range(vertical_subregions):
            y = i*region_size  + region_size // 2
            x = j*region_size  + region_size // 2
            subregion_indices.append((x, y))

    subregion_indices = np.array(subregion_indices)    
    V_x = np.array(V_x)
    V_y = np.array(V_y)
    return subregion_indices, V_x, V_y
    
def calculate_optical_flow_with_LK_for_corners(name_image1='Car1.jpg', name_image2='Car2.jpg', region_size=15, r=np.array([5]), c=np.array([5])):
    
    c = c[r >= (region_size//2)]
    r = r[r >= (region_size//2)]

    r = r[c >= (region_size//2)]
    c = c[c >= (region_size//2)]

    I_t0, I_t1 = load_images(name_image1, name_image2)
    
    I_x, I_y, I_t = calculate_derivatives(I_t0, I_t1)

    sub_I_x, sub_I_y, sub_I_t = calculate_subregions_for_corners(I_x, I_y, I_t, r, c, region_size)

    V_x, V_y = calculate_flow_vectors(sub_I_x, sub_I_y, sub_I_t)
        
    subregion_indices = np.array((c, r)).T    
    
    return subregion_indices, V_x, V_y




if __name__ == '__main__':

    original_image = cv2.imread('./images/Car2.jpg')
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    subregion_indices, V_x, V_y = calculate_optical_flow_with_LK()
    plt.figure()
    plt.imshow(original_image)
    plt.quiver(subregion_indices[:,0], subregion_indices[:,1], V_x, V_y, angles='xy', scale_units='xy', scale=0.1)
    plt.show()

   



    









