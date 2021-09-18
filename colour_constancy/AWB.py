import numpy as np
import cv2
import matplotlib.pyplot as plt

def grey_world(img):
    img = ((img * (img.mean() / img.mean(axis=(0, 1))))
             .clip(0, 255).astype(int))
    return img


if __name__ == '__main__':
    original_img = cv2.imread('./colour_constancy/awb.jpg')
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    img_after_gw = grey_world(original_img)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(img_after_gw)
    plt.title("Colour corrected")
    plt.axis('off')
    plt.show()