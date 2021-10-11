# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import random
from keypoint_matching import calculate_keypoint_matching


def AffineTransformation(stackofpoints):
    
    
    return


def RANSAC(corr, thresh):
    maxInliers = []
    finalH = None
    #Repeat N times (selected 1000)
    for i in range(1000):
        #Pick P matches at random from the total set of matches T (selected 6)
        for i in range(6):
            corr[i] = corr[random.randrange(0, len(corr))]
            if i == 0: 
                continue
            elif i == 1:
                stackofpoints = np.vstack((corr[0], corr[1]))
            else:
                stackofpoints = np.vstack((stackofpoints, corr[i]))
        P = AffineTransformation(stackofpoints)
    return 5, 6


if __name__ == '__main__':
    
    #load images
    img1 = cv.imread('boat1.pgm')
    img2 = cv.imread('boat2.pgm')
    
    #make them gray, to be more precise delete the 2 of 3 chanels 
    #because images are already gray "pgm" files
    img1_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    
    #set the estimation threshold
    est_thres = 0.6
    
    #calculate matching keypoints
    kp1, kp2, des1, des2, matches = calculate_keypoint_matching(img1_gray, img2_gray)
    
    templist = []
    for match in matches:
        (x1, y1) = kp1[match.queryIdx].pt
        (x2, y2) = kp2[match.trainIdx].pt
        templist.append([x1, y1, x2, y2])
    correspondence = np.matrix(templist)
    
    #call RANSAC function
    P, Inliers = RANSAC(correspondence, est_thres)
    