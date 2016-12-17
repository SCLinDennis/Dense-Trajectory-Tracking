#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:32:12 2016

@author: DennisLin
"""
import numpy as np
import cv2
#people detection 
#from __future__ import print_function
from imutils.object_detection import non_max_suppression
import argparse
#import imutils
import time

def selectwindow(record_info_clear):

    checkpoint = 0 #if 0 then a, if 1 then b
    person_A = []
    person_B = []
    if record_info_clear[0,1] < 130: #initialize
        person_A_temp = record_info_clear[0]
        person_A.append(person_A_temp)
        person_A_info = np.array(person_A)
        person_B_info = np.array([])
        checkpoint = 0
    elif record_info_clear[0,1] > 130:
        person_B_temp = record_info_clear[0]
        person_B.append(person_B_temp)
        person_B_info = np.array(person_B)
        person_A_info = np.array([])
        checkpoint = 1
    [row_record_info_clear,col_record_info_clear] = np.shape(record_info_clear)

#    index_A = 0
#    index_B = 0 
   
    
    for i in range(row_record_info_clear-1):
        if (record_info_clear[i+1,0] != record_info_clear[i,0]):
            if checkpoint == 0:
                center_dist = np.linalg.norm((record_info_clear[i+1,1]+record_info_clear[i+1,3])-(record_info_clear[i,1]+record_info_clear[i,3]))
                if center_dist <= 5:
                    person_A_temp = record_info_clear[i+1]
                    person_A.append(person_A_temp)
                    person_A_info = np.array(person_A)
#                    index_A = index_A + 1
                    checkpoint = 0
                elif (center_dist > 5) and (center_dist <= 40):
                    record_info_clear[i+1,1:5]  = record_info_clear[i,1:5]                    
                else:
                    person_B_temp = record_info_clear[i+1]
                    person_B.append(person_B_temp)
                    person_B_info = np.array(person_B)
                    checkpoint = 1 
            else: 
                center_dist = np.linalg.norm((record_info_clear[i+1,1]+record_info_clear[i+1,3])-(record_info_clear[i,1]+record_info_clear[i,3]))
                if  center_dist <= 5:
                    person_B_temp = record_info_clear[i+1]
                    person_B.append(person_B_temp)
                    person_B_info = np.array(person_B)
                    checkpoint = 1 
                elif (center_dist > 5) and (center_dist <= 40):
                    record_info_clear[i+1,1:5]  = record_info_clear[i,1:5]                     
                else:
                    person_A_temp = record_info_clear[i+1]
                    person_A.append(person_A_temp)
                    person_A_info = np.array(person_A)
#                    index_A = index_A + 1
                    checkpoint = 0           
        else:# same frame
            if checkpoint == 0:
                person_B_temp = record_info_clear[i+1]
                person_B.append(person_B_temp)
                person_B_info = np.array(person_B) 
                checkpoint = 1
            else:
                person_A_temp = record_info_clear[i+1]
                person_A.append(person_A_temp)
                person_A_info = np.array(person_A) 
                checkpoint = 0            
    return (person_A_info, person_B_info)
AnswerA = np.array([])
AnswerB = np.array([])
#record_info_clear = unique_rows(record_info)
record_info_clear = np.load('/Users/DennisLin/tmp/record_info_clear.npy')
(AnswerA,AnswerB) = selectwindow(record_info_clear)      


camera = cv2.VideoCapture('/Users/DennisLin/Videos/April30_2sentence1.mp4')

#load the features
row = np.load('/Users/DennisLin/feats_npy_file/{April30_2sentence1.mpg}_out_features.npy')
[width, length] = row.shape
width = int(width)
length = int(length)


## initialize the HOG descriptor/person detector
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# take first frame of the video
(grabbed,frame_old) = camera.read()
#rame = frame_old[:,:,:]
frame = frame_old[:,:,:]
r = 400.0 / frame.shape[1]
dim = (400, int(frame.shape[0] * r))
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#(rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32,32), scale=1.05)
#rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
#rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
#pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
frame_num = 0
while(camera.isOpened()):
    ret, frame = camera.read()
    r = 400.0 / frame.shape[1]
    dim = (400, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#    track_window2 = np.array([[pick[i,0], pick[i,1], 75,250] for i in range(pick.shape[0])])
#    cv2.rectangle(frame, (x1,y1), (x1+w1,y1+h1), (0,255,0),2)
#    for (xA, yA, xB, yB) in pick:
#        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    for i in range(AnswerA.shape[0]) :
        if frame_num == AnswerA[i,0]:
            cv2.rectangle(frame, (AnswerA[i,1],AnswerA[i,2]), (AnswerA[i,3], AnswerA[i,4]), (0, 255, 0), 2)
    for i in range(AnswerB.shape[0]) :
        if frame_num == AnswerB[i,0]:
            cv2.rectangle(frame, (AnswerB[i,1],AnswerB[i,2]), (AnswerB[i,3], AnswerB[i,4]), (0, 0, 255), 2)            
    cv2.putText(frame, "Frame_index: {}".format(frame_num), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
            #    for e in range(width):
#        if row[e,0] == frame_num:
#            (x,y) = (row[e,1], row[e,2])
#            for (x_rect, y_rect, w_rect, h_rect) in rects:
#                if (x_rect < int(x*r)) and (int(x*r) < (x_rect+w_rect)) and (y_rect < int(y*r)) and (int(y*r) < (y_rect+h_rect)):
#                    cv2.circle(orig_resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)                    

#    cv2.rectangle(frame, (365, 130), (470, 400), (0, 255, 255), 2)

#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_num +=1

camera.release()
cv2.destroyAllWindows()      
