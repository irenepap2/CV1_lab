import cv2
import matplotlib.pyplot as plt
import numpy as np
from myPSNR import myPSNR

def denoise( image, kernel_type, **kwargs):
    ksize = kwargs['kernel_size']
   

    if kernel_type == 'box':
        imOut = cv2.blur(image, (ksize,ksize))
    elif kernel_type == 'median':
        imOut = cv2.medianBlur(image, ksize)
    elif kernel_type == 'gaussian':
        imOut = cv2.GaussianBlur(image, (ksize,ksize), sigmaX=kwargs['sigma'])
    else:
        print('Operation not implemented')

    # cv2.imshow('image',image)
    # cv2.imshow('imOut',imOut)
    # cv2.waitKey(0)
    return imOut


if __name__ == '__main__':
    image_dir = './Image_enhancement/images/'
    image_name = ['image1_saltpepper.jpg','image1_gaussian.jpg']


    kernel_type = ['box', 'median', 'gaussian']
    title = ['3x3', '5x5', '7x7']
    ksize = [3,5,7]

    PSNR_results = np.zeros((4,3))

    for image_index in range(2):
        input_image = cv2.imread(image_dir + image_name[image_index], 0)
        for kernel_index in range(2):
            fig, ax = plt.subplots(1,3,figsize=(13,3))
            for i in range(3):
                imOut = denoise(input_image, kernel_type[kernel_index], kernel_size=ksize[i])
                ax[i].imshow(imOut, cmap='gray')
                ax[i].set_title(title[i])
                PSNR_results[image_index*2 + kernel_index, i] = myPSNR(input_image, imOut)
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
            plt.savefig('denoised_'+image_name[image_index][:-4]+'_'+kernel_type[kernel_index]+'.png',transparent=True)
    print(PSNR_results)


    PSNR_results_gaussian = np.zeros((2,4))
    title_g = ['3x3', '5x5', '7x7', '9x9']
    ksize_g = [3,5,7,9]
    

    for image_index in range(2):
        input_image = cv2.imread(image_dir + image_name[image_index], 0)
        fig, ax = plt.subplots(1,4,figsize=(16,3))
        for i in range(4):
            imOut = denoise(input_image, kernel_type[2], kernel_size=ksize_g[i], sigma=(ksize_g[i]-1)/6)
            PSNR_results_gaussian[image_index, i] = myPSNR(input_image, imOut)

            ax[i].imshow(imOut, cmap='gray')
            ax[i].set_title(title_g[i])
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        plt.savefig('denoised_'+image_name[image_index][:-4]+'_'+kernel_type[2]+'.png',transparent=True)
    print(PSNR_results_gaussian)
    plt.show()

#  PSNR TABLE
# [16.43044422 15.78900821 15.42222213] SP-BOX
# [16.12840336 15.7622603  15.43774736] SP-MEDIAN
# [20.3606286  19.18655533 18.40717152] G -BOX
# [20.30698232 19.31360284 18.54007006] G -MEDIAN

# [23.61764504 23.61764504 23.61764504] SP-GAUSSIAN
# [27.6767747  27.6767747  27.6767747 ] G -GAUSSIAN