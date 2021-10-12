import matplotlib.pyplot as plt
import cv2
import numpy as np
from keypoint_matching import calculate_keypoint_matching
from RANSAC import *

if __name__ == '__main__':
    
    #load images
    image_right = cv2.imread('boat2.pgm')
    image_left = cv2.imread('boat1.pgm')
    
    #make them gray, to be more precise delete the 2 of 3 chanels 
    #because images are already gray "pgm" files
    image_right_gray = cv2.cvtColor(image_right,cv2.COLOR_BGR2GRAY)
    image_left_gray = cv2.cvtColor(image_left,cv2.COLOR_BGR2GRAY)
    
    #convert BGR to RGB
    image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)
    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)

    #set the estimation threshold
    est_thres = 10
    
    #calculate matching keypoints
    kp1, kp2, des1, des2, matches = calculate_keypoint_matching(image_right_gray, image_left_gray)
    
    templist = []
    for match in matches:
        (x1, y1) = kp1[match.queryIdx].pt
        (x2, y2) = kp2[match.trainIdx].pt
        templist.append([x1, y1, x2, y2])
    correspondence = np.matrix(templist)
    
    #call RANSAC function
    P = RANSAC(correspondence, est_thres)
    
    
    
    h, w = image_right_gray.shape
    corners = np.array([[0, 0, w, w], [0, h, 0, h], [1, 1, 1, 1]])
    warp_corners = (P @ corners)[0:2]

    bottom = np.floor(warp_corners[1].max()).astype(int)
    right = np.floor(warp_corners[0].max()).astype(int)
    warped_image_right = cv2.warpAffine(image_right, P[0:2,:], (right, bottom))

    cv2.imwrite('warped_image_right.jpg', warped_image_right)
    
    bottom = max(bottom, image_left.shape[0])
    right = max(right, image_left.shape[1])
    
    new_size = (bottom, right, 3)

    final_image_left_and_right = np.zeros(new_size)
    final_image_left_and_right[:warped_image_right.shape[0], :warped_image_right.shape[1], :] = warped_image_right
    final_image_left_and_right[:image_left.shape[0], :image_left.shape[1], :] = image_left
    final_image_left_and_right /= 255
    
    # BGR to RGB
    # final_image_left_and_right = final_image_left_and_right[:, :, ::-1]
    # image_right = image_right[:, :, ::-1]
    # image_left = image_left[:, :, ::-1]
    
    
    fig=plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image_left)
    plt.title('left')
    plt.subplot(1, 3, 2)
    plt.imshow(image_right)
    plt.title('right')
    plt.subplot(1, 3, 3)
    plt.imshow(final_image_left_and_right)
    plt.title('stiched image')
    plt.show()