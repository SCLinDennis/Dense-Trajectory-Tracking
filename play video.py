#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:23:24 2016

@author: DennisLin
"""

import numpy as np 
import argparse
#import imutils
import time
import cv2
from imutils.object_detection import non_max_suppression




ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",default="/Users/DennisLin/Videos/April30_2sentence1.mp4", help="Path to the file")
#ap.add_argument("-f", "--features", default='/Users/DennisLin/feats_npy_file/April30_2sentence2.mpg_out_features.npy', help="Path to the file")
#ap.add_argument("-a", "--min-area", type=int, default=300, help="minimum area size")
args = vars(ap.parse_args())


# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])



#cap = cv2.VideoCapture('vtest.avi')
frame_num = 0
#index_A = 0
#index_B = 0

while(camera.isOpened()):
    ret, frame = camera.read()
    r = 400.0 / frame.shape[1]
    dim = (400, int(frame.shape[0] * r))
    
    resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#    resized_frame = resized_frame[:,0:resized_frame.shape[1]-50,:]
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame_num == AnswerA[frame_num,0]:
        cv2.rectangle(resized_frame,(AnswerA[frame_num,1],AnswerA[frame_num,2]),(AnswerA[frame_num,3], AnswerA[frame_num,4]), (0, 255, 0), 2)
#        index_A = index_A + 1 
    if frame_num == AnswerB[frame_num,0]:
        cv2.rectangle(resized_frame,(AnswerB[frame_num,1],AnswerB[frame_num,2]),(AnswerB[frame_num,3], AnswerB[frame_num,4]), (0, 0, 255), 2)
    cv2.imshow('frame',resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_num += 1
camera.release()
cv2.destroyAllWindows()