# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:20:12 2016

@author: Shih-Chen Lin
"""

#mean shift 
import numpy as np
import cv2
#people detection 
#from __future__ import print_function
from imutils.object_detection import non_max_suppression
import argparse
#import imutils
import time

def euclidean (track_windowA, track_windowB):
                #2*4            2*4
    for i in range(2):
        dist0 = np.zeros(2)
        dist0[i] = np.linalg.norm(track_windowA[0]-track_windowB[i])
    if dist0[1]<dist0[0]:
        track_windowA = np.array(track_windowA[1],track_windowA[0])
    else:
        track_windowA = track_windowA
    return track_windowA;

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="D://Videos/April30_2sentence1.mpg", help="the path for video")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])


#camera = cv2.VideoCapture('D://Videos/slow_traffic_small.mp4')

# take first frame of the video
(grabbed,frame) = camera.read()
(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.25)
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
#rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)


###################mean_shift###########
# setup initial location of window
# r,h,c,w - region of image
#           simply hardcoded the values    
track_window2 = np.array([[pick[i,0], pick[i,1], 75,300] for i in range(pick.shape[0])])

# set up the ROI for tracking
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#threshold the HSV image to get certain color
mask = cv2.inRange(hsv_roi, np.array((0., 0.,0.)), np.array((50.,50.,30.)))
#for black (0, 0, 0) & (29, 30, 30)
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
frame_num = 0;
while(1):
    #bool ret if is 1, read correctly
    grabbed ,frame = camera.read()
    if grabbed == True:
        if (frame_num % 30 == 0) and (frame_num != 0):
            print(1)           
            (new_rects, new_weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.25)
            new_rects2 = np.array([[new_x, new_y, new_x + new_w, new_y + new_h] for (new_x, new_y, new_w, new_h) in new_rects])
            new_pick = non_max_suppression(new_rects2, probs=None, overlapThresh=0.65)
            track_window_tmp = np.array([[new_pick[j,0], new_pick[j,1], 75,300] for j in range(new_pick.shape[0])])
            track_window2 = euclidean(track_window_tmp,track_window2)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        #grabbed2 = grabbed.copy()
        arr1  =   np.array(track_window2[0])
        arr2  =   np.array(track_window2[1])    
        track_window3 = tuple(arr1)
        track_window4 = tuple(arr2)
        grabbed, track_window3 = cv2.meanShift(dst, track_window3, term_crit)
        grabbed, track_window4 = cv2.meanShift(dst, track_window4, term_crit)
        x1,y1,w1,h1 = track_window3
        x2,y2,w2,h2 = track_window4        
            
        # Draw it on image
        cv2.rectangle(frame, (x1,y1), (x1+w1,y1+h1), (0,255,0),2)
        cv2.rectangle(frame, (x2,y2), (x2+w2,y2+h2), (0,0,255),2)                
        cv2.imshow('img2',frame)
#        cv2.imshow(dst)

        k = cv2.waitKey(30) & 0xff
        if k == ord("q"):
            break
    else:
        break
        print('false')
    frame_num +=1

cv2.destroyAllWindows()
camera.release()
