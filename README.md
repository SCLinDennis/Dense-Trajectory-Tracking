# 林詩宸(Shih-Chen Lin)  
# BiiC Lab 
# National Hsing Hua University, HsinChu, Taiwan
 

## Senior Project / Learning Human Emotion from Actor's Interaction in Drama Rehearsal Clips

## Overview
The project is related to 
> The goal of this project is to learn human emotion from actors' behavior. Specifically, we will first use DenseTrajectory optimized by Wang et al, 2011 to extract features from people, and then train the model using SVM classifier. 

## Implement
* ```HOG_latest.py```: Used to detect the people in the frame 
* ```selectwindow_checker.py```: Used to detemine the bounding boxes we detect in the previous function are corresponded to whom.
* ```occulusion.py```: To make the condequences more robust, we should deal with the case that two people in the video passed wach others, also named "Occlusion".
