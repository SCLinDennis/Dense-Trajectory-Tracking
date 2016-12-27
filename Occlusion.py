# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 12:45:48 2016

@author: user
"""

import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import argparse
#import imutils
import time
from sklearn import svm
import matplotlib.pyplot as plt
import pdb
import sys

def SVM_validation(A_feature_ori, B_feature_ori, A_feature_test_ori, B_feature_test_ori, feature_start, feature_end, K):
    if(feature_end > 437 or feature_start < 0):
        print "[ValueError] Invalid feature range !"
        sys.exit()

    A_dim = A_feature_ori.shape[0]
    B_dim = B_feature_ori.shape[0]
    A_test_dim = A_feature_test_ori.shape[0]
    B_test_dim = B_feature_test_ori.shape[0]
    
    # label append on the last column(person A will be 0, person B will be 1)
    # that means the dimension of each row will be 
    # 1*(feature_end-feature_start +1)
    y_A = [0] * (A_dim)
    y_B = [1] * (B_dim)
    y_A_test = [0] * (A_test_dim)
    y_B_test = [1] * (B_test_dim)
    A_feature = A_feature_ori[0:A_dim, feature_start:feature_end]
    B_feature = B_feature_ori[0:B_dim, feature_start:feature_end]
#    pdb.set_trace()
    A_feature_test = A_feature_test_ori[0:A_test_dim, feature_start:feature_end]
    B_feature_test = B_feature_test_ori[0:B_test_dim, feature_start:feature_end]    
    
    A_feature = np.c_[A_feature, y_A]#add one column
    B_feature = np.c_[B_feature, y_B]
    A_feature_test = np.c_[A_feature_test, y_A_test]
    B_feature_test = np.c_[B_feature_test, y_B_test]

    X = np.append(A_feature, B_feature, axis=0) # mix them 
    X_test = np.append(A_feature_test, B_feature_test, axis=0)
    # shuffle the training data
    np.random.shuffle(X)
    np.random.shuffle(X_test)
    

    # Then, we use K-fold cross-validation to estimate our accuracy
    each_part = X.shape[0]/K
    each_part_test = X_test.shape[0]/K
    tmp_accu = 0
    total_accu = 0
#    total_accu_a = 0
#    total_accu_b = 0
    for fold in range(K):
        if(fold != K-1):
            valid_data = X[each_part*fold : each_part*(fold+1) , :]
            test_data = X_test[each_part_test*fold : each_part_test*(fold+1),: ]
            training_data = np.append(X[0 : each_part*fold, :], X[each_part*(fold+1): X.shape[0], :], axis=0)
        else:
            valid_data = X[each_part*fold : , :]
            test_data = X_test[each_part_test :,:]
            training_data = X[0 : each_part*fold, :]
        row_dim = training_data.shape[1]
        feature = training_data[:, 0:row_dim-1]
        label = training_data[:, row_dim-1]
        classifier = svm.SVC(kernel='linear', C = 1.0)
        classifier.fit(feature, label)

#        w = classifier.coef_[0]# what is this for
        correct = 0
#        valid_data_num = valid_data.shape[0]
        test_data_num = test_data.shape[0]
        
#        for valid in range(valid_data_num):
#            pre = classifier.predict(valid_data[valid, 0:row_dim-1])
#            if int(pre) == int(valid_data[valid][row_dim-1]):
#                correct = correct + 1
        for test in range(test_data_num):
            pre = classifier.predict(test_data[test, 0:row_dim-1])
            if int(pre) == int(test_data[test][row_dim-1]):
                correct = correct + 1
        tmp_accu = (correct/float(test_data_num))*100
        total_accu = total_accu + tmp_accu
        print "[Fold " + str(fold+1) + "/" + str(K) + "] Accuracy = " + str(tmp_accu) + "%"
        
    print "Total Accuracy = " + str(total_accu/K) + "%"
    if total_accu/K > 50:
        change_or_not = False
    elif total_accu/K < 50:
        change_or_not = True
    return change_or_not


camera = cv2.VideoCapture('/Users/DennisLin/Videos/April30_2sentence1.mp4')
# take first frame of the video
(grabbed,frame_old) = camera.read()
#rame = frame_old[:,:,:]
frame = frame_old[:,:,:]
r = 400.0 / frame.shape[1]
dim = (400, int(frame.shape[0] * r))

#load the features
feature = np.load('/Users/DennisLin/feats_npy_file/{April30_2sentence1.mpg}_out_features.npy')
AnswerA = np.load('/Users/DennisLin/tmp/AnswerA-2.npy')
AnswerB = np.load('/Users/DennisLin/tmp/AnswerB-2.npy')
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
            feature_train = feature[frame_index2[1],0:244] #HOG_features         
            for j in range(AnswerA.shape[0]):# j == frame_num from Answer
                feature_train_tmp_tmp = feature_train[np.where(j == feature_train[:,0]),:]
                if len(feature_train_tmp_tmp[0]) != 0:      
                    feature_train_tmp = feature_train_tmp_tmp[0,:,:]
                    for k in range(feature_train_tmp.shape[0]):
                        if (AnswerA[j,0] == feature_train_tmp[k,0]) and (AnswerA[j,1] < feature_train_tmp[k,2]*r) and (feature_train_tmp[k,2]*r < AnswerA[j,3]) and (AnswerA[j,2] < feature_train_tmp[k,1]*r) and (feature_train_tmp[k,1]*r < AnswerA[j,4]):      
                            Answer_A_temp = feature_train_tmp[k,0:244]
                            Answer_A_array.append(Answer_A_temp)
                            Answer_A_feat_train = np.array(Answer_A_array)
                        if (AnswerB[j,0] == feature_train_tmp[k,0]) and  (AnswerB[j,1] < feature_train_tmp[k,2]*r) and (feature_train_tmp[k,2]*r < AnswerB[j,3]) and (AnswerB[j,2] < feature_train_tmp[k,1]*r) and (feature_train_tmp[k,1]*r < AnswerB[j,4]):
                            Answer_B_temp = feature_train[k,0:244]
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
            feature_testing = feature[feature_index2[1],0:244] #HOG_features 
            for m in range(recog_A.shape[0]):
                feature_testing_tmp_tmp = feature_testing[np.where(recog_A[m,0] == feature_testing[:,0]),:]
                if len(feature_testing_tmp_tmp[0]) != 0:  
                    feature_testing_tmp = feature_testing_tmp_tmp[0,:,:]
                    for n in range(feature_testing_tmp.shape[0]):  
                        if recog_A[m,0] == feature_testing_tmp[n,0] and recog_A[m,1] < feature_testing_tmp[n,2]*r and feature_testing_tmp[n,2]*r < recog_A[m,3] and recog_A[m,2] < feature_testing_tmp[n,1]*r and feature_testing_tmp[n,1]*r < recog_A[m,4]:
                            recog_A_temp = feature_testing_tmp[n,0:244]
                            recog_A_array.append(recog_A_temp)
                            recog_A_feat = np.array(recog_A_array)
                        if recog_B[m,0] == feature_testing_tmp[n,0] and recog_B[m,1] < feature_testing_tmp[n,2]*r and feature_testing_tmp[n,2]*r < recog_B[m,3] and recog_B[m,2] < feature_testing_tmp[n,1]*r and feature_testing_tmp[n,1]*r < recog_B[m,4]:
                            recog_B_temp = feature_testing_tmp[n,0:244]
                            recog_B_array.append(recog_B_temp)
                            recog_B_feat = np.array(recog_B_array)
                    ###################SVM recognition#########################
                        if (SVM_validation(Answer_A_feat_train, Answer_B_feat_train, recog_A_feat, recog_B_feat, 10, 244, 10) == True ):
                            AnswerA[i-100:,:],AnswerB[i-100,:]=AnswerB[i-100:,:].copy(),AnswerA[i-100,:].copy()
                        
                    ###########################################################
                    ##################Change or not chaange####################
                        
                        
                    ###########################################################
print("done")

