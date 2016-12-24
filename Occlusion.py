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

camera = cv2.VideoCapture('/Users/DennisLin/Videos/April30_2sentence1.mp4')
# take first frame of the video
(grabbed,frame_old) = camera.read()
#rame = frame_old[:,:,:]
frame = frame_old[:,:,:]
r = 400.0 / frame.shape[1]
dim = (400, int(frame.shape[0] * r))

#load the features
feature = np.load('/Users/DennisLin/feats_npy_file/{April30_2sentence1.mpg}_out_features.npy')
AnswerA = np.load('/Users/DennisLin/tmp/AnswerA.npy')
AnswerB = np.load('/Users/DennisLin/tmp/AnswerB.npy')
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
recog_A_feat = np.array(recog_A_array)
recog_B_array = []
recog_B_feat = np.array(recog_B_array)
#stage_list = [] #0 train;   1 dont do ;   2 test 
#stage_array = np.array(stage_list)

occlusion_time = 0 
frame_accum = 0

for i in range(row_A):# i == frame_num
    ######################colecting the first training data############################## 
    if np.linalg.norm((AnswerA[i,1]+AnswerA[i,3])-(AnswerB[i,1]+AnswerB[i,3])) <= 70: 
        if occlusion_time == 0:
            occlusion_time += 1
            train_limit = i-1
            frame_index1 = np.where(feature[:,0] <=train_limit)
            ###########take HOG feature#######################
            frame_index2 = np.where(feature[frame_index1,0] >= 600)
            feature_train = feature[frame_index2[1],0:137] #HOG_features         
            for j in range(AnswerA.shape[0]):# j == frame_num from AnswerA
                feature_train_tmp = feature_train[np.where(j == feature_train[:,0]),:]
                for k in range(feature_train_tmp.shape[0]):
                    if AnswerA[j,0] == feature_train_tmp[k,0] and AnswerA[j,1] < feature_train_tmp[k,2]*r and feature_train_tmp[k,2]*r < AnswerA[j,3] and AnswerA[j,2] < feature_train_tmp[k,1]*r and feature_train_tmp[k,1]*r < AnswerA[j,4]:
                        Answer_A_temp = feature_train_tmp[k,10:137]
                        Answer_A_array.append(Answer_A_temp)
                        Answer_A_feat_train = np.array(Answer_A_array)
                    if AnswerB[j,0] == feature_train_tmp[k,0] and AnswerB[j,1] < feature_train_tmp[k,2]*r and feature_train_tmp[k,2]*r < AnswerB[j,3] and AnswerB[j,2] < feature_train_tmp[k,1]*r and feature_train_tmp[k,1]*r < AnswerB[j,4]:
                        Answer_B_temp = feature_train[k,10:137]
                        Answer_B_array.append(Answer_B_temp)
                        Answer_B_feat_train = np.array(Answer_B_array)
            ####################Train SVM######################
                            
            
            ###################################################
        elif occlusion_time > 0:
            frame_accum = 0
            occlusion_time += 1
    ##################################################################################### 
    elif occlusion_time > 0 and np.linalg.norm((AnswerA[i,1]+AnswerA[i,3])-(AnswerB[i,1]+AnswerB[i,3])) >= 70:
        frame_accum += 1 
        if frame_accum == 100:           
            recog_A = AnswerA[i-100:i]
            recog_B = AnswerB[i-100:i]
            feature_index1 = np.where(feature[:,0] <= i-1) 
            feature_index2 = np.where(feature[feature_index1,0] >= i-100)
            feature_testing = feature[feature_index2[1],0:137] #HOG_features 
            for m in range(recog_A.shape[0]):
                feature_testing_tmp = feature_testing[np.where(recog_A[m,0] == feature_testing[:,0]),:]
                for n in range(feature_testing_tmp.shape[0]):
                    if recog_A[m,0] == feature_testing_tmp[n,0] and recog_A[m,1] < feature_testing_tmp[n,2]*r and feature_testing_tmp[n,2]*r < recog_A[m,3] and recog_A[m,2] < feature_testing_tmp[n,1]*r and feature_testing_tmp[n,1]*r < recog_A[m,4]:
                        recog_A_temp = feature_testing_tmp[n,10:137]
                        recog_A_array.append(recog_A_temp)
                        recog_A_feat = np.array(recog_A_array)
                    if recog_B[m,0] == feature_testing_tmp[n,0] and recog_B[m,1] < feature_testing_tmp[n,2]*r and feature_testing_tmp[n,2]*r < recog_B[m,3] and recog_B[m,2] < feature_testing_tmp[n,1]*r and feature_testing_tmp[n,1]*r < recog_B[m,4]:
                        recog_B_temp = feature_testing_tmp[n,10:137]
                        recog_B_array.append(recog_B_temp)
                        recog_B_feat = np.array(recog_B_array)
                    ###################SVM recognition#########################
                        
                        
                    ###########################################################
                    ##################Change or not chaange####################
                        
                        
                    ###########################################################


