# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:40:25 2016

@author: Shih-Chen Lin
"""

import numpy as np 
import argparse
import imutils
import time
import cv2
from imutils.object_detection import non_max_suppression

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",default="/Users/DennisLin/Videos/April30_2sentence1.mp4", help="Path to the file")
ap.add_argument("-a", "--min-area", type=int, default=300, help="minimum area size")
args = vars(ap.parse_args())


#load the features
row = np.load('/Users/DennisLin/feats_npy_file/{April30_2sentence1.mpg}_out_features.npy')
[width, length] = row.shape
width = int(width)
length = int(length)

# Parameters for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
  
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors to draw line
color = np.random.randint(0,255,(100,3))

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])
 
 
#########optical flow##################
 
# Take first frame and find corners in it
(grabbed, old_frame) = camera.read()
#if not grabbed :
#    break
my_height = 400.0
r = my_height / old_frame.shape[1]
dim = (int(my_height), int(old_frame.shape[0] * r))
old_frame2 = cv2.resize(old_frame, dim, interpolation = cv2.INTER_AREA)

old_gray = cv2.cvtColor(old_frame2, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame2)


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#"count" is used to indicate the index of the frmae
count = 0;
while True:
    # grab the current frame
    (grabbed, frame_tmp) = camera.read()
    frame_gray_tmp = cv2.cvtColor(frame_tmp, cv2.COLOR_BGR2GRAY)

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
    if args.get("video") and not grabbed:
		break

    #change the size of the frame_gray
    r2 = my_height / frame_gray_tmp.shape[1]
    dim2 = (int(my_height), int(frame_gray_tmp.shape[0] * r2))
    frame_gray = cv2.resize(frame_gray_tmp, dim2, interpolation = cv2.INTER_AREA)
    
    #change the size of the frame
    r3 = my_height / frame_tmp.shape[1]
    dim3 = (int(my_height), int(frame_tmp.shape[0] * r3))
    frame = cv2.resize(frame_tmp, dim3, interpolation = cv2.INTER_AREA)
    
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):#list (index, (object))
        a,b = new.ravel()
        c,d = old.ravel()
        #cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(frame,(a,b),5,color[i].tolist(),-1)    

    ##############hog detection###############
    orig_resized = frame.copy()    
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
    for (x_rect, y_rect, w_rect, h_rect) in rects:
                cv2.rectangle(orig_resized, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 0, 255), 2)

    
#    for e in range(width):
#        if row[e,0] == count:
#            (x,y) = (row[e,1], row[e,2])
#            #cv2.circle(resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)
#            for (x_rect, y_rect, w_rect, h_rect) in rects:
#                if (x_rect < int(x*r3)) and (int(x*r3) < (x_rect+w_rect)) and (y_rect < int(y*r3)) and (int(y*r3) < (y_rect+h_rect)):
#                    #cv2.rectangle(orig_resized, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 0, 255), 2)
#                    cv2.circle(orig_resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)                    
#            rects_nms = np.array([[x_rect, y_rect, x_rect + w_rect, y_rect + h_rect] for (x_rect, y_rect, w_rect, h_rect) in rects])
#            pick = non_max_suppression(rects_nms, probs=None, overlapThresh=1.2)
#            for (xA, yA, xB, yB) in pick:
#                if (xA < int(x*r3)) and (int(x*r3) < (xA+xB)) and (yA < int(y*r3)) and (int(y*r3) < (yA+yB)):
#                    #cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
#                    cv2.circle(frame, (int(x*r3), int(y*r3)), int(1),(0, 255, 255), 2)
#    for (xA, yA, xB, yB) in pick:
#                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
#    for e in range(width):
    frame_index = np.where(row[:,0]==count)
#    frame_index = np.array(frame_index)
#    for t in frame_index:
#        test = np.array([[row[t,1] ,row[t,2]])
    test = row[frame_index,1:3]
    test = test[0,:,:]
            
##        if row[e,0] == count:
#    test = np.array([row[frame_index,1] row[frame_index,2]])
#            #cv2.circle(resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)
    for (x_rect, y_rect, w_rect, h_rect) in rects:
        for  (x_point, y_point) in test:
            if (x_rect < int(x_point*r3)) and (int(x_point*r3) < (x_rect+w_rect)) and (y_rect < int(y_point*r3)) and (int(y_point*r3) < (y_rect+h_rect)):
                    #cv2.rectangle(orig_resized, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 0, 255), 2)
                cv2.circle(orig_resized, (int(x_point*r), int(y_point*r)), int(1),(0, 255, 255), 2)  
          
            
#            if (x_rect < int(x*r3)) and (int(x*r3) < (x_rect+w_rect)) and (y_rect < int(y*r3)) and (int(y*r3) < (y_rect+h_rect)):
#                    #cv2.rectangle(orig_resized, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 0, 255), 2)
#                cv2.circle(orig_resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)                    
#            rects_nms = np.array([[x_rect, y_rect, x_rect + w_rect, y_rect + h_rect] for (x_rect, y_rect, w_rect, h_rect) in rects])
#            pick = non_max_suppression(rects_nms, probs=None, overlapThresh=1.2)
#            for (xA, yA, xB, yB) in pick:
#                if (xA < int(x*r3)) and (int(x*r3) < (xA+xB)) and (yA < int(y*r3)) and (int(y*r3) < (yA+yB)):
#                    #cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
#                    cv2.circle(frame, (int(x*r3), int(y*r3)), int(1),(0, 255, 255), 2)
#    for (xA, yA, xB, yB) in pick:
#                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    img = cv2.add(frame,mask)
    #img2 = cv2.add(img, )
    	# show the frame to our screen
    cv2.imshow("After_NMS_Frame", img)
    cv2.imshow("Before_NMS_Fram", orig_resized)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    count +=1

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
