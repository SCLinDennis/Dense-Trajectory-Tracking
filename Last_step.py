#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:19:49 2017

@author: DennisLin
"""
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import cv2

import argparse
import imutils
import time


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",default="/Users/DennisLin/Videos/April30/April30_2sentence1.mp4", help="Path to the file")
ap.add_argument("-s", "--start", type=int, help="start frame")
ap.add_argument("-e", "--end", type=int, help="end_frame")
args = vars(ap.parse_args())


# start the file video stream thread and allow the buffer to
# start to fill
print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

# start the FPS timer
fps = FPS().start()

start = args["start"]
end = args["end"]
AnswerA = np.load('/Users/DennisLin/AnswerA-21s_v2.npy')
AnswerB = np.load('/Users/DennisLin/AnswerB-21s_v2.npy')

AnswerA[start:end,:],AnswerB[start:end,:]=AnswerB[start:end,:].copy(),AnswerA[start:end,:].copy()

frame_num = 0

# loop over frames from the video file stream
while fvs.more():
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale (while still retaining 3
	# channels)
	frame = fvs.read()
	r = 400.0 / frame.shape[1]
	dim = (400, int(frame.shape[0] * r))
	frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#	frame = imutils.resize(frame, width=450)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = np.dstack([frame, frame, frame])
	if frame_num == AnswerA[frame_num,0]:
		cv2.rectangle(frame,(AnswerA[frame_num,1],AnswerA[frame_num,2]),(AnswerA[frame_num,3], AnswerA[frame_num,4]), (0, 255, 0), 2)
	if frame_num == AnswerB[frame_num,0]:
		cv2.rectangle(frame,(AnswerB[frame_num,1],AnswerB[frame_num,2]),(AnswerB[frame_num,3], AnswerB[frame_num,4]), (0, 0, 255), 2)
	# display the size of the queue on the frame
	cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
	cv2.putText(frame, "Frame_index: {}".format(frame_num), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	# show the frame and update the FPS counter
	
	cv2.imshow("Frame", frame)
	frame_num += 1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break     
	fps.update()
  
  
#while(camera.isOpened()):
#    ret, frame = camera.read()
#    r = 400.0 / frame.shape[1]
#    dim = (400, int(frame.shape[0] * r))
#    
#    resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#    resized_frame = resized_frame[:,0:resized_frame.shape[1]-50,:]
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    if frame_num == AnswerA[frame_num,0]:
#        cv2.rectangle(resized_frame,(AnswerA[frame_num,1],AnswerA[frame_num,2]),(AnswerA[frame_num,3], AnswerA[frame_num,4]), (0, 255, 0), 2)
##        index_A = index_A + 1 
#    if frame_num == AnswerB[frame_num,0]:
#        cv2.rectangle(resized_frame,(AnswerB[frame_num,1],AnswerB[frame_num,2]),(AnswerB[frame_num,3], AnswerB[frame_num,4]), (0, 0, 255), 2)
#    cv2.putText(resized_frame, "Frame_index: {}".format(frame_num), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#    cv2.imshow('frame',resized_frame)
#    
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#    frame_num += 1
fps.stop()

np.save('/Users/DennisLin/AnswerA-21o_last.npy', AnswerA)
np.save('/Users/DennisLin/AnswerB-21o_last.npy', AnswerB)    
#camera.release()
cv2.destroyAllWindows()
fvs.stop()