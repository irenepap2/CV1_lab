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


    return imOut


if __name__ == '__main__':
    image_dir = './Image_enhancement/images/'
    original_image_name = 'image1.jpg'
    image_name = ['image1_saltpepper.jpg','image1_gaussian.jpg']
    
    original_image = cv2.imread(image_dir + original_image_name, 0)


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
                PSNR_results[image_index*2 + kernel_index, i] = myPSNR(original_image, imOut)
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
            plt.savefig('denoised_' + image_name[image_index][:-4] + '_' + kernel_type[kernel_index] + '.png',
                        transparent=True)
    print(PSNR_results)


    # Titles kernel_sizes and sigmas for the plots of Gaussian filter
    title_g = ['3x3', '5x5', '7x7', '15x15']
    ksize_g = [3, 5, 7, 15]
    sigmas = [0.5, 1, 5, 10]
    
    # Take the gaussian noise image
    image_index = 1
    input_image = cv2.imread(image_dir + image_name[image_index], 0)


    PSNR_results_gaussian = np.zeros((4,4))
    for s_i in range(4):
        for k_i in range(4):
            imOut = denoise(input_image, kernel_type[2], kernel_size=ksize_g[k_i], sigma=sigmas[s_i])
            PSNR_results_gaussian[s_i, k_i] = myPSNR(original_image, imOut)   
        
    print(PSNR_results_gaussian)


    
    print('ax shape: ',ax.shape)
    for s_i in range(4):
        for i in range(4):
            fig, ax = plt.subplots(1,1,figsize=(5,5))
            imOut = denoise(input_image, kernel_type[2], kernel_size=ksize_g[i], sigma=sigmas[s_i])

            ax.imshow(imOut, cmap='gray')
            ax.set_title('Kernel size:'+title_g[i]+',  Sigma:'+str(sigmas[s_i]))
            plt.subplots_adjust(left=0.08, bottom=0.05, right=0.95, top=0.95)
            plt.savefig('denoised_' + image_name[image_index][:-4] + '_' + kernel_type[2] + '_' + str(s_i) + 'x' + str(i) + '.png',
                        transparent=True)
    plt.show()