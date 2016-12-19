# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 20:36:14 2016

@author: user
"""

import numpy as np 
import argparse
#import imutils
import time
import cv2
from imutils.object_detection import non_max_suppression

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",default="/Users/DennisLin/Videos/April30_2sentence1.mp4", help="Path to the file")
ap.add_argument("-f", "--features", default='/Users/DennisLin/feats_npy_file/{April30_2sentence1.mpg}_out_features.npy', help="Path to the file")
ap.add_argument("-a", "--min-area", type=int, default=300, help="minimum area size")
args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#load the features
row = np.load(args["features"])

#row = np.load('/Users/DennisLin/feats_npy_file/{April30_2sentence1.mpg}_out_features.npy')
[width, length] = row.shape
width = int(width)
length = int(length)


# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

# Take first frame and find corners in it
(ret, old_frame) = camera.read()
r = 400.0 / old_frame.shape[1]
dim = (400, int(old_frame.shape[0] * r))
resized_old_frame = cv2.resize(old_frame, dim, interpolation = cv2.INTER_AREA)
resized_old_gray = cv2.cvtColor(resized_old_frame, cv2.COLOR_BGR2GRAY)
(rects_old, weights_old) = hog.detectMultiScale(resized_old_frame, winStride=(8, 8),
		padding=(8, 8), scale=1.05)
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    
    
#def selectwindow(record_info_clear):
#    if record_info_clear[0,1] < 130: #initialize
#        person_A_temp = record_info_clear[0]
#        person_A.append(person_A_temp)
#        person_A_info = np.array(person_A)
#        person_B_info = np.array([])
#    elif record_info_clear[0,1] > 130:
#        person_B_temp = record_info_clear[0]
#        person_B.append(person_B_temp)
#        person_B_info = np.array(person_B)
#        person_A_info = np.array([])
#    [row_record_info_clear,col_record_info_clear] = np.shape(record_info_clear)
#
#    index_A = 0
#    index_B = 0    
#    
#    for i in range(row_record_info_clear-1):
#        if (record_info_clear[i+1,0] != record_info_clear[i,0]):
#            if np.linalg.norm(record_info_clear[i＋1,1:5]-record_info_clear[i,1:5]) <= 30):
#                person_A_temp = record_info_clear[i+1]
#                person_A.append(person_A_temp)
#                person_A_info = np.array(person_A)
#                index_A = index_A + 1
#            else:
#                person_B_temp = record_info_clear[i+1]
#                person_B.append(person_B_temp)
#                person_B_info = np.array(person_B)
#        else:# same frame
#            if (record_info_clear[i+1,1] < record_info_clear[i,1]):
#                person_A_temp = record_info_clear[i+1]
#                person_A.append(person_A_temp)
#                person_A_info = np.array(person_A)
#                index_A = index_A + 1
#            else:
#                person_B_temp = record_info_clear[i+1]
#                person_B.append(person_B_temp)
#                person_B_info = np.array(person_B)    
#    

#a matrix to record the human box
record = []
record_info = np.array([])

person_A = []
person_A_info = np.array(person_A)
person_B = []
person_B_info = np.array(person_B)

count = 0

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
    frame_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    orig_resized = resized.copy()
    (rects, weights) = hog.detectMultiScale(resized, winStride=(8, 8),
		padding=(8, 8), scale=1.05)
    
    frame_index = np.where(row[:,0]==count)
#    frame_index = np.array(frame_index)
#    for t in frame_index:
#        test = np.array([[row[t,1] ,row[t,2]])
    test = row[frame_index,1:3]
    test = test[0,:,:]
    rects_nms = np.array([[x_rect, y_rect, x_rect + w_rect, y_rect + h_rect] for (x_rect, y_rect, w_rect, h_rect) in rects])
    pick = non_max_suppression(rects_nms, probs=None, overlapThresh=0.65)

    for (x1, y1, x2, y2) in pick:
        for  (x_point, y_point) in test:
            if (x1 < int(x_point*r)) and (int(x_point*r) < (x2)) and (y1 < int(y_point*r)) and (int(y_point*r) < (y2)):
                    #cv2.rectangle(orig_resized, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 0, 255), 2)
                cv2.circle(orig_resized, (int(x_point*r), int(y_point*r)), int(1),(0, 255, 255), 2) 
                cv2.rectangle(orig_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                record_temp = [count,x1,y1,x2,y2]
                record.append(record_temp)
                record_info = np.array(record)
  
    cv2.putText(orig_resized, "Frame_index: {}".format(count), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#    for (x_rect, y_rect, w_rect, h_rect) in rects:
#        for f in range(width):
#            if row[f,0] == count:
#                (x1,y1) = (row[f,1], row[f,2])
#                if (x_rect < int(x1*r)) and (int(x1*r) < (x_rect+w_rect)) and (y_rect < int(y1*r)) and (int(y1*r) < (y_rect+h_rect)):
#                    cv2.rectangle(orig_resized, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 0, 255), 2)
#                    record_temp = [count,x_rect,y_rect,x_rect+w_rect,y_rect+h_rect]
#                    record.append(record_temp)
#                    record_info = np.array(record)
        
    
    #ditinguish   130 is the middle
####################################################################################################
    #deal with the record_info to delete same rectangle
    #use the function unique_rows
    
#    record_info_clear = unique_rows(record_info)
####################################################################################################
    
####################################################################################################
    
    
    
####################################################################################################    

########after make a clean matrix of (frame,x,y), doing the distinguish#############################        
#    if record_info_clear[0,1] < 130:
#        person_A_temp = record_info_clear[0]
#        person_A.append(person_A_temp)
#        person_A_info = np.array(person_A)
#    elif record_info_clear[0,1] > 130:
#        person_B_temp = record_info_clear[0]
#        person_B.append(person_B_temp)
#        person_B_info = np.array(person_B)
#    [row_record_info_clear,col_record_info_clear] = np.shape(record_info_clear)
#
#    index_A = 0
#    index_B = 0    
#    
#    for i in range(row_record_info_clear-1):
#        if (record_info_clear[i+1,0] != reco1rd_info_clear[i,0]):
#            if (record_info_clear[i+1,1] - person_A_info[index_A,1] <= 20):
#                person_A_temp = record_info_clear[i+1]
#                person_A.append(person_A_temp)
#                person_A_info = np.array(person_A)
#                index_A = index_A + 1
#            else:
#                person_B_temp = record_info_clear[i+1]
#                person_B.append(person_B_temp)
#                person_B_info = np.array(person_B)
#        else:
#            if (record_info_clear[i+1,1] < record_info_clear[i,1]):
#                person_A_temp = record_info_clear[i+1]
#                person_A.append(person_A_temp)
#                person_A_info = np.array(person_A)
#                index_A = index_A + 1
#            else:
#                person_B_temp = record_info_clear[i+1]
#                person_B.append(person_B_temp)
#                person_B_info = np.array(person_B)
####################################################################################################

    
    #plot the trajectory features in the rect
#    for e in range(width):
#        if row[e,0] == count:
#            (x,y) = (row[e,1], row[e,2])
#            for (x_rect, y_rect, w_rect, h_rect) in rects:
#                if (x_rect < int(x*r)) and (int(x*r) < (x_rect+w_rect)) and (y_rect < int(y*r)) and (int(y*r) < (y_rect+h_rect)):
#                    cv2.circle(orig_resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)                    
#            rects_nms = np.array([[x_rect, y_rect, x_rect + w_rect, y_rect + h_rect] for (x_rect, y_rect, w_rect, h_rect) in rects])
#            pick = non_max_suppression(rects_nms, probs=None, overlapThresh=0.65)
#            for (xA, yA, xB, yB) in pick:
#                if (xA < int(x*r)) and (int(x*r) < (xB)) and (yA < int(y*r)) and (int(y*r) < (yB)):
#                    cv2.circle(resized, (int(x*r), int(y*r)), int(1),(0, 255, 255), 2)
#    #plot rect after NMS
#    for (xA, yA, xB, yB) in pick:
#                cv2.rectangle(resized, (xA, yA), (xB, yB), (0, 255, 0), 2)
                
#    cv2.imshow("After_NMS_Frame", resized)
    cv2.imshow("After_NMS_Fram", orig_resized)
    
    
    #store old rect coordinate    
    (rects_old, weights_old) = (rects, weights)
#    pick_old = pick

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    count +=1
np.save('record_info.npy', record_info)
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
