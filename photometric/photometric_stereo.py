import numpy as np
import cv2
import os
from utils import *
from estimate_alb_nrm import estimate_alb_nrm
from check_integrability import check_integrability
from construct_surface import construct_surface

print('Part 1: Photometric Stereo\n')

def photometric_stereo(image_dir='./SphereGray5/', isRGB = False):

    if not isRGB:
        # obtain many images in a fixed view under different illumination
        print('Loading images...\n')
        [image_stack, scriptV] = load_syn_images(image_dir)
        [h, w, n] = image_stack.shape
        print('Finish loading %d images.\n' % n)

        # compute the surface gradient from the stack of imgs and light source mat
        print('Computing surface albedo and normal map...\n')
        [albedo, normals] = estimate_alb_nrm(image_stack, scriptV, True)

        # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
        print('Integrability checking\n')
        [p, q, SE] = check_integrability(normals)

        threshold = 0.005
        print('Number of outliers: %d\n' % np.sum(SE > threshold))
        SE[SE <= threshold] = float('nan') # for good visualization

        # compute the surface height
        height_map = construct_surface( p, q )

        # show results
        show_results(albedo, normals, height_map, SE)
    
    else:
        albedos = None
        for i in range(3):

            print('Loading images...\n')
            [image_stack, scriptV] = load_syn_images(image_dir, channel=i)
            [h, w, n] = image_stack.shape

            if albedos is None:
               albedos = np.zeros([h, w, 3], dtype=np.float64)
               normals_all_channels = np.zeros([h, w, 3, 3], dtype=np.float64)
               height_map_all_channels = np.zeros([h, w, 3], dtype=np.float64)
               SE_all_channels = np.zeros([h, w, 3], dtype=np.float64)
               
            print('Finish loading %d images.\n' % n)

            # compute the surface gradient from the stack of imgs and light source mat
            print('Computing surface albedo and normal map...\n')
            [albedo, normals] = estimate_alb_nrm(image_stack, scriptV, False)

            # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
            print('Integrability checking\n')
            [p, q, SE] = check_integrability(normals)

            threshold = 0.005
            print('Number of outliers: %d\n' % np.sum(SE > threshold))
            SE[SE <= threshold] = float('nan') # for good visualization

            # compute the surface height
            height_map = construct_surface( p, q )

            albedos[:, :, i] = albedo
            normals_all_channels[:, :, :, i] = normals
            height_map_all_channels[:, :, i] = height_map
            SE_all_channels[:, :, i] = SE

        # show results
        show_results_RGB(albedos, normals_all_channels, height_map_all_channels, SE_all_channels)

## Face
def photometric_stereo_face(image_dir='./yaleB02/'):
    [image_stack, scriptV] = load_face_images(image_dir)
    [h, w, n] = image_stack.shape
    print('Finish loading %d images.\n' % n)
    print('Computing surface albedo and normal map...\n')
    albedo, normals = estimate_alb_nrm(image_stack, scriptV, False)

    # integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
    print('Integrability checking')
    p, q, SE = check_integrability(normals)

    threshold = 0.005
    print('Number of outliers: %d\n' % np.sum(SE > threshold))
    SE[SE <= threshold] = float('nan') # for good visualization

    # compute the surface height
    height_map = construct_surface( p, q , 'column')

    # show results
    show_results(albedo, normals, height_map, SE)
    
if __name__ == '__main__':
    # photometric_stereo('./photometric/photometrics_images/SphereGray25/', isRGB=False)
    photometric_stereo_face('./photometric/photometrics_images/yaleB02/')
