import numpy as np
import cv2
import rgbConversions
from visualize import *

def ConvertColourSpace(input_image, colourspace):
    '''
    Converts an RGB image into a specified color space, visualizes the
    color channels and returns the image in its new color space.

    Colorspace options:
      opponent
      rgb -> for normalized RGB
      hsv
      ycbcr
      gray

    P.S: Do not forget the visualization part!
    '''

    # Convert the image into double precision for conversions
    input_image = input_image.astype(np.float32)
    
    if colourspace.lower() == 'opponent':
        # fill in the rgb2opponent function
        new_image = rgbConversions.rgb2opponent(input_image)

    elif colourspace.lower() == 'rgb':
        # fill in the rgb2opponent function
        new_image = rgbConversions.rgb2normedrgb(input_image)

    elif colourspace.lower() == 'hsv':
        # use built-in function from opencv
        new_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)
        new_image[:, :, 0] = new_image[:, :, 0] * (255/360)
        new_image[:, :, 1] = new_image[:, :, 1] * 255
        
        pass

    elif colourspace.lower() == 'ycbcr':
        # use built-in function from opencv
        new_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2YCrCb)
        
        # Swap Cr and Cb
        # new_image_y = new_image[:, :, 0]
        new_image_Cr = np.array(new_image[:, :, 1])
        new_image_Cb = np.array(new_image[:, :, 2])

        new_image[:, :, 1] = new_image_Cb
        new_image[:, :, 2] = new_image_Cr

        cv2.normalize(new_image,  new_image, 0, 255, cv2.NORM_MINMAX)
        pass

    elif colourspace.lower() == 'gray':
        # fill in the rgb2opponent function
        new_image = rgbConversions.rgb2grays(input_image)

    else:
        print('Error: Unknown colorspace type [%s]...' % colourspace)
        new_image = input_image


    visualize(new_image)

    return new_image


if __name__ == '__main__':
    # Replace the image name with a valid image
    img_path = './colourspace/kouzes.png'
    # Read with opencv
    I = cv2.imread(img_path)
    
    # Convert from BGR to RGB
    # This is a shorthand.
    I = I[:, :, ::-1]


    out_img = ConvertColourSpace(I, 'opponent')
    out_img = ConvertColourSpace(I, 'rgb')
    out_img = ConvertColourSpace(I, 'hsv')
    out_img = ConvertColourSpace(I, 'ycbcr')
    out_img = ConvertColourSpace(I, 'gray')
    
    print('Converted')
    plt.show()