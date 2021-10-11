# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import random
from keypoint_matching import calculate_keypoint_matching


def AffineTransformation(stackofpoints):
    A = []
    b = []
    for i in range(len(stackofpoints)):
        A.append([stackofpoints[i, 0], stackofpoints[i, 1], 0, 0, 1, 0])
        A.append([0, 0, stackofpoints[i, 0], stackofpoints[i, 1], 0, 1])
        b.append(stackofpoints[i, 2])
        b.append(stackofpoints[i, 3]) 
    
    x = np.linalg.pinv(A).dot(b)

    affine = np.array([[x[0], x[1], x[4]], [x[2], x[3], x[5]], [0, 0, 1]])

    return affine


def RANSAC(corr, thresh):
    N = 50
    P = 6
    max_inliers = -1
    #Repeat N times (selected 1000)
    for i in range(N):
        #Pick P matches at random from the total set of matches T (selected 6)
        for i in range(P):
            corr[i] = corr[random.randrange(0, len(corr))]
            if i == 0: 
                continue
            elif i == 1:
                stackofpoints = np.vstack((corr[0], corr[1]))
            else:
                stackofpoints = np.vstack((stackofpoints, corr[i]))
        affine = AffineTransformation(stackofpoints)

        im1_points = []
        im2_points = stackofpoints[:,2:4].T
        print(im2_points.shape)
        
        for i in range(len(stackofpoints)):
            im1_points.append([stackofpoints[i, 0], stackofpoints[i, 1], 1])

        # transform first image points
        transformed_points = (affine @ np.array(im1_points).T)[:2]
        print(transformed_points.shape)
        
        print(np.square(transformed_points - im2_points))
        print(np.sum(np.square(transformed_points - im2_points)))
        print('\n')
        distance = np.sqrt(np.sum(np.square(transformed_points - im2_points), axis=0))
        inliers = np.shape(distance[distance<=200])[1]

        if (inliers > max_inliers):
            max_inliers = inliers
            best_points = np.copy(stackofpoints)
            best_solution = np.copy(affine)

            print(best_points)
            print(best_solution)
    
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
    