import matplotlib.pyplot as plt
import numpy as np
from  matplotlib.colors import LinearSegmentedColormap


def visualize(input_image):
    # Fill in this function. Remember to remove the pass command
    
    # Convert image numbers from float32 to uint8 in order to visualize them
    input_image = input_image.astype(np.uint8)
    
    fig = plt.figure()

    number_of_channels = input_image.shape[2]

    cmap_red_green=LinearSegmentedColormap.from_list('rg',["g", "k", "r"], N=256) 
    cmap_yellow_blue=LinearSegmentedColormap.from_list('yb',["b", "k", "y"], N=256) 

    if number_of_channels == 3:

        # Show the converted image
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(input_image)
        ax1.set_title('Converted Image')

        # Show the first channel
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(input_image[:, :, 0], cmap='gray', vmin=0 ,vmax=255)
        ax2.set_title('Channel 1')

        # Show the second channel
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(input_image[:, :, 1], cmap='gray', vmin=0 ,vmax=255)
        ax3.set_title('Channel 2')

        # Show the third channel
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(input_image[:, :, 2], cmap='gray', vmin=0 ,vmax=255)
        ax4.set_title('Channel 3')

    else:

        # Show the converted image using lightness
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(input_image[:, :, 0], cmap='gray', vmin=0, vmax=255)
        ax1.set_title('Lightness')

        # Show the converted image using average
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(input_image[:, :, 1], cmap='gray', vmin=0, vmax=255)
        ax2.set_title('Average')

        # Show the converted image using luminocity
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(input_image[:, :, 2], cmap='gray', vmin=0, vmax=255)
        ax3.set_title('Luminocity')

        # Show the converted image using openCV
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(input_image[:, :, 3], cmap='gray', vmin=0, vmax=255)
        ax4.set_title('openCV')

    plt.tight_layout()
    plt.show(block=False)
