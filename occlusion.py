# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 12:45:48 2016

@author: user
"""

import numpy as np
import cv2
#people detection 
#from __future__ import print_function
from imutils.object_detection import non_max_suppression
import argparse
#import imutils
import time

camera = cv2.VideoCapture('D://senior/CCL/video/April30/April30_2sentence1.mpg')
# take first frame of the video
(grabbed,frame_old) = camera.read()
#rame = frame_old[:,:,:]
frame = frame_old[:,:,:]
r = 400.0 / frame.shape[1]
dim = (400, int(frame.shape[0] * r))

#load the features
feature = np.load('D://senior/CCL/{April30_2sentence1.mpg}_out_features.npy')
AnswerA = np.load('D://senior/CCL/special_topic/AnswerA.npy')
AnswerB = np.load('D://senior/CCL/special_topic/AnswerB.npy')
[row, col] = feature.shape
row = int(row)
col = int(col)
[row_A,col_A] = AnswerA.shape
[row_B,col_B] = AnswerB.shape
Answer_A_array = []
Answer_A_feat_train = np.array(Answer_A_array)
Answer_B_array = []
Answer_B_feat_train = np.array(Answer_B_array)

recog_A_array = []
recog_B_array = []



occlusion_time = 0 
frame_accum = 0

for i in range(row_A):
   ######################colecting the first training data############################## 
    if occlusion_time ==0:
        if np.linalg.norm(AnswerA[i,1]-AnswerB[i,1]) <= 70:
            occlusion_time += 1
            train_limit = i-1
            frame_index = np.where(feature[:,0] <=train_limit)
            ###########take HOG feature#######################
            feature_selection = feature[frame_index,10:137]
            for j in range(feature_selection.shape[0]):
                if feature_selection[j,0] > 600:
                    for k in range(train_limit):
                        if AnswerA[k,0] == feature_selection[j,0] and AnswerA[k,0] >= 600 and AnswerA[k,1] < feature_selection[j,2]*r and feature_selection[j,2]*r < AnswerA[k,3] and AnswerA[k,2] < feature_selection[j,1]*r and feature_selection[j,1]*r < AnswerA[k,4]:
                            Answer_A_temp = feature_selection[j]
                            Answer_A_array.append(Answer_A_temp)
                            Answer_A_feat_train = np.array(Answer_A_array)
                        if AnswerB[k,0] == feature_selection[j,0] and AnswerB[k,0] >= 600 and AnswerB[k,1] < feature_selection[j,2]*r and feature_selection[j,2]*r < AnswerB[k,3] and AnswerB[k,2] < feature_selection[j,1]*r and feature_selection[j,1]*r < AnswerB[k,4]:
                            Answer_B_temp = feature_selection[j]
                            Answer_B_array.append(Answer_B_temp)
                            Answer_B_feat_train = np.array(Answer_B_array)
            ####################Train SVM######################
                            
            
            ###################################################
                
    ##################################################################################### 
    elif occlusion_time > 0 and np.linalg.norm(AnswerA[i,1]-AnswerB[i,1]) >= 70:
        frame_accum += 1
        if frame_accum == 100:
            recog_A = AnswerA[i-99:i]
            recog_B = AnswerB[i-99:i]
            for m in range(row):
                for n in range(frame_accum):
                    if recog_A[n,0] == feature[m,0] and recog_A[n,1] < feature[m,2]*r and feature[m,2]*r < recog_A[n,3] and recog_A[n,2] < feature[m,1]*r and feature[m,1]*r < recog_A[n,4]:
                        recog_A_temp = feature[m]
                        recog_A_array.append(recog_A_temp)
                        recog_A_feat = np.array(recog_A_array)
                    if recog_B[n,0] == feature[m,0] and recog_B[n,1] < feature[m,2]*r and feature[m,2]*r < recog_B[n,3] and recog_B[n,2] < feature[m,1]*r and feature[m,1]*r < recog_B[n,4]:
                        recog_B_temp = feature[m]
                        recog_B_array.append(recog_B_temp)
                        recog_B_feat = np.array(recog_B_array)
                    ###################SVM recognition#########################
                        
                        
                    ###########################################################
                    ##################Change or not chaange####################
                        
                        
                    ###########################################################
        
    
                    
    elif occlusion_time > 0 and np.linalg.norm(AnswerA[i,1]-AnswerB[i,1]) <= 70:
        frame_accum = 0
        occlusion_time += 1
        
