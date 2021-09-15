import numpy as np
import cv2
import os
from utils import *
from check_integrability import check_integrability



def estimate_alb_nrm( image_stack, scriptV, shadow_trick=True):
    
    # COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
    # INPUT:
    # image_stack : the images of the desired surface stacked up on the 3rd dimension
    # scriptV : matrix V (in the algorithm) of source and camera information
    # shadow_trick: (true/false) whether or not to use shadow trick in solving linear equations
    # OUTPUT:
    # albedo : the surface albedo
    # normal : the surface normal

    
    h,w,n = image_stack.shape
    
    
    # # create arrays for 
    # # albedo (1 channel)
    # # normal (3 channels)
    albedo = np.zeros([h, w])
    normal = np.zeros([h, w, 3])

    
    """
    ================
    Your code here()
    ================
    for each point in the image array
        stack image values into a vector i
        construct the diagonal matrix scriptI
        solve scriptI * scriptV * g = scriptI * i to obtain g for this point
        albedo at this point is |g|
        normal at this point is g / |g|
    """

    for x in range(w):
        for y in range(h):
            i = image_stack[x,y,:]
            
            if shadow_trick:
                scriptI = np.diag(i)
            else:
                scriptI = np.eye(n)

            g, _, _, _ = np.linalg.lstsq(np.matmul(scriptI,scriptV),np.matmul(scriptI,i),rcond=None)
            
            albedo[x,y] = np.linalg.norm(g)
            normal[x,y,:] = g/albedo[x,y]

    return albedo, normal
    
if __name__ == '__main__':
    n = 5


    
    # image_stack, scriptV = load_syn_images(image_dir='./lab1/photometric/photometrics_images/SphereGray5/')

    
    albedo, normals = estimate_alb_nrm( image_stack, scriptV, shadow_trick=True)
    check_integrability(normals)