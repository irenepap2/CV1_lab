import numpy as np
import cv2
def myPSNR( orig_image, approx_image ):
    m, n = orig_image.shape

    orig_image = orig_image.astype(np.float32)
    approx_image = approx_image.astype(np.float32)

    MSE = np.sum(((approx_image - orig_image)**2)) / (m*n)

    I_max = orig_image.max()

    PSNR = 20 * np.log10(I_max / np.sqrt(MSE))

    return PSNR

if __name__ == '__main__':
    orig_image_path = './Image_enhancement/images/image1.jpg'
    approx_image_path = './Image_enhancement/images/image1_gaussian.jpg'

    orig_image = cv2.imread(orig_image_path,0)
    approx_image = cv2.imread(approx_image_path,0)
    
    img_PSNR = myPSNR(orig_image, approx_image)

    print('PSNR:', img_PSNR)

# img/img_saltpepper PSNR: 16.107930175560774 
# img/img_gaussian PSNR: 20.583544923613655 