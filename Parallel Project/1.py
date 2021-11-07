# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:12:11 2019

@author: SHIKHAR GUPTA
"""

import cv2 as cv

img = cv.imread('Capture.JPG')
cv.imshow('Out',img)
cv.waitKey(0)
cv.destroyAllWindows()