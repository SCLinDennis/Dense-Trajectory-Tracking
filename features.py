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

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",default="D://Videos/April30_2sentence1.mpg", help="Path to the file")
ap.add_argument("-a", "--min-area", type=int, default=300, help="minimum area size")
args = vars(ap.parse_args())

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
    r = 800.0 / frame.shape[1]
    dim = (800, int(frame.shape[0] * r))

    # perform the actual resizing of the image and show it
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    #cv2.imshow("resized", resized)  
    for a in range(i):
        if row[a,0] == count:
            (x,y) = (row[a,1], row[a,2])
            cv2.circle(resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)
    	# show the frame to our screen
    cv2.imshow("Frame", resized)
    key = cv2.waitKey(3) & 0xFF
    if key == ord("q"):
        break
    count +=1

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()