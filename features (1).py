# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:40:25 2016

@author: user
"""

import numpy as np 
import argparse
#import imutils
import time
import cv2
from imutils.object_detection import non_max_suppression

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",default="D://Videos/April30_2sentence1.mpg", help="Path to the file")
ap.add_argument("-a", "--min-area", type=int, default=300, help="minimum area size")
args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#load the features
row = np.load('C://Users/user/feats_npy_file/{April30_2sentence1.mpg}_out_features.npy')
[i, j] = row.shape
i = int(i)
j = int(j)
#for a in range(i):
#    for b in range(j):
#        if j>=10 && j<=24:
#            for c in range(29):
#                Trajectory[a][c]
#            
#        print row[a,b],
#    print '\n'

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

count = 0;
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
    if args.get("video") and not grabbed:
		break
    #frame = imutils.resize(frame, width=500)
    r = 400.0 / frame.shape[1]
    dim = (400, int(frame.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    orig_resized = resized.copy()
    (rects, weights) = hog.detectMultiScale(resized, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
    for (x_rect, y_rect, w_rect, h_rect) in rects:
                cv2.rectangle(orig_resized, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 0, 255), 2)
    #cv2.imshow("resized", resized)  
    for a in range(i):
        if row[a,0] == count:
            (x,y) = (row[a,1], row[a,2])
            #cv2.circle(resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)
            for (x_rect, y_rect, w_rect, h_rect) in rects:
                #cv2.rectangle(orig_resized, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 0, 255), 2)                
                if (x_rect < int(x*r)) and (int(x*r) < (x_rect+w_rect)) and (y_rect < int(y*r)) and (int(y*r) < (y_rect+h_rect)):
                    cv2.circle(orig_resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)                    
            rects_nms = np.array([[x_rect, y_rect, x_rect + w_rect, y_rect + h_rect] for (x_rect, y_rect, w_rect, h_rect) in rects])
            pick = non_max_suppression(rects_nms, probs=None, overlapThresh=0.65)
            for (xA, yA, xB, yB) in pick:
                #cv2.rectangle(resized, (xA, yA), (xB, yB), (0, 255, 0), 2)
                if (xA < int(x*r)) and (int(x*r) < (xB)) and (yA < int(y*r)) and (int(y*r) < (yB)):
                    cv2.circle(resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)
    for (xA, yA, xB, yB) in pick:
                cv2.rectangle(resized, (xA, yA), (xB, yB), (0, 255, 0), 2)
    	# show the frame to our screen
    cv2.imshow("After_NMS_Frame", resized)
    cv2.imshow("Before_NMS_Fram", orig_resized)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    count +=1

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()