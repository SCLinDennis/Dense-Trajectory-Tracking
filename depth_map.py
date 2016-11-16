# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:52:35 2016

@author: user
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

imgR = cv2.imread('D://Videos/tsukuba_l (1).png',0)
imgL = cv2.imread('D://Videos/tsukuba_r.png',0)

stereo = cv2.StereoBM(1, 16, 15)
disparity = stereo.compute(imgL, imgR)

plt.imshow(disparity,'gray')
plt.show()
