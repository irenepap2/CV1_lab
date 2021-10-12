import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import random
from keypoint_matching import calculate_keypoint_matching



def my_warp(original_img, transformation_params):
    h, w = original_img.shape
    
    #first we create a grid to perform the coordinate wise transformation on
    initial_coordinates = np.indices((h, w)).T
    initial_coordinates = initial_coordinates.reshape(-1,2)

    M = np.array([[transformation_params[0, 0],transformation_params[0, 1]], [transformation_params[1, 0], transformation_params[1, 1]]]) # trafo matritrafo_params
    translation = np.array([transformation_params[0, 2], transformation_params[1, 2]])

    #now we apply the transformation on the original grid
    new_coordinates = np.array(np.around((initial_coordinates @ np.linalg.inv(M).T) - translation[None,:]), dtype=np.int32)
    
    # new_coordinates = (transformation_params@np.vstack((initial_coordinates.T, np.ones((1,initial_coordinates.shape[0])))))[0:2]
    # new_coordinates = new_coordinates.T.astype(np.int32)
    #find minimum and maximum x and y values to be able to fit the warped image into a picture frame
    min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
    for i in range(len(new_coordinates)):
        if new_coordinates[i][0] < min_x:
            min_x = new_coordinates[i][0]
        if new_coordinates[i][0] > max_x:
            max_x = new_coordinates[i][0]
        if new_coordinates[i][1] < min_y:
            min_y = new_coordinates[i][1]
        if new_coordinates[i][1] > max_y:
            max_y = new_coordinates[i][1]

    #offset the coords to have min_x value of 0 and min_y value of 0
    for j in range(len(new_coordinates)):
        new_coordinates[j][0] = new_coordinates[j][0] - min_x
        new_coordinates[j][1] = new_coordinates[j][1] - min_y

    
    new_im = np.ones((max_x-min_x+1, max_y-min_y+1)) * -1
    new_im[new_coordinates[:,0], new_coordinates[:,1]] = original_img[initial_coordinates[:,0], initial_coordinates[:,1]]
    new_h, new_w = new_im.shape
    for i in range(new_h):
        for j in range(new_w):
            if new_im[i,j]<0:
                end = j+10
                if end > new_w-1:
                    end= new_w-1
                num_of_missing = sum(new_im[i,j:end]<0)
                if j>0 and num_of_missing<10:
                    new_im[i,j] = new_im[i,j-1] 

    return new_im



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
    N = 1000
    P = 10
    max_inliers = -1
    #Repeat N times (selected 1000)
    for i in range(N):
        for i in range(P):
            corr[i] = corr[random.randrange(0, len(corr))]
            if i == 0: 
                continue
            elif i == 1:
                stackofpoints = np.vstack((corr[0], corr[1]))
            else:
                stackofpoints = np.vstack((stackofpoints, corr[i]))
        affine = AffineTransformation(stackofpoints)

        im2_points = corr[:,2:4].T
        im1_points = []
        for i in range(len(corr)):
            im1_points.append([corr[i,0], corr[i, 1], 1])

        # transform first image points
        transformed_points = (affine @ np.array(im1_points).T)[:2]
        
       
        distance = np.sqrt(np.sum(np.square(transformed_points - im2_points), axis=0))
        inliers = np.shape(distance[distance<=thresh])[1]

        if (inliers > max_inliers):
            max_inliers = inliers
            best_solution = np.copy(affine)
            

          
    print('MAX inliers found:',max_inliers)
    return best_solution


if __name__ == '__main__':
    
    #load images
    image_right = cv.imread('boat1.pgm')
    image_left = cv.imread('boat2.pgm')
    
    #make them gray, to be more precise delete the 2 of 3 chanels 
    #because images are already gray "pgm" files
    image_right_gray = cv.cvtColor(image_right,cv.COLOR_BGR2GRAY)
    image_left_gray = cv.cvtColor(image_left,cv.COLOR_BGR2GRAY)
    
    image_right = image_right[:, :, ::-1]
    image_left = image_left[:, :, ::-1]

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
    
    
   
    
    # transformed_points = (P @ np.array(im1_points).T)[:2]
    h, w = image_right_gray.shape
    image1_cvwarp = cv.warpAffine(image_right_gray,P[0:2,:],(w,h))
    image1_mywarp = my_warp(image_right_gray, P)
    
    h, w = image_left_gray.shape
    new_P = np.linalg.inv(P)
    image2_cvwarp = cv.warpAffine(image_left_gray, new_P[0:2,:], (w,h))
    image2_mywarp = my_warp(image_left_gray, new_P)

    print()
    fig=plt.figure(figsize=(20,30))
    plt.subplot(2, 3, 1)
    plt.imshow(image_right)
    plt.axis('off')
    plt.title('Original image 1', fontsize=22)
    plt.subplot(2, 3, 2)
    plt.imshow(image1_cvwarp, vmin=0, vmax=255, cmap='gray')
    plt.axis('off')
    plt.title('image 1 to image 2 using openCV warp function', fontsize=22)
    plt.subplot(2, 3, 3)
    plt.imshow(image1_mywarp, vmin=0, vmax=255, cmap='gray')
    plt.axis('off')
    plt.title('image 1 to image 2 using our warp function', fontsize=22)
    plt.subplot(2, 3, 4)
    plt.imshow(image_left)
    plt.axis('off')
    plt.title('Original image 2', fontsize=22)
    plt.subplot(2, 3, 5)
    plt.imshow(image2_cvwarp, vmin=0, vmax=255, cmap='gray')
    plt.axis('off')
    plt.title('image 2 to image 1 using openCV warp function', fontsize=22)
    plt.subplot(2, 3, 6)
    plt.imshow(image2_mywarp, vmin=0, vmax=255, cmap='gray')
    plt.axis('off')
    plt.title('image 2 to image 1 using our warp function', fontsize=22)
    
    plt.show()
