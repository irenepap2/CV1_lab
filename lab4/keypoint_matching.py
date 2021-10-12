# -*- coding: utf-8 -*-
import cv2 as cv
import random

def calculate_keypoint_matching(img1, img2):
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    return kp1, kp2, des1, des2, matches

if __name__ == '__main__':
    
    #load images
    img1 = cv.imread('left.jpg')
    img2 = cv.imread('right.jpg')
    
    #make them gray, to be more precise delete the 2 of 3 chanels 
    #because images are already gray "pgm" files
    img1_gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    
    #calculate matching keypoints
    kp1, kp2, des1, des2, matches = calculate_keypoint_matching(img1, img2)
    
    #plot a random subset
    sample_matches = random.sample(matches, 100)
    match_img = cv.drawMatches(img1, kp1, img2, kp2, sample_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite('./Subset_matches.jpg', match_img)
    cv.imshow('Matching subset of size 10', match_img)
    cv.waitKey(0)

   



    








