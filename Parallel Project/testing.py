from __future__ import (division, absolute_import, print_function, unicode_literals)
import cv2 as cv
#import numpy as np
#import scipy.ndimage
#import copy

a = [[2,3],
     [4,5],
     [7,8]]
print(sum(map(sum,a)))
print(pow(5,2))
print(len(a[1][:]))#column
print(len(a[:]))#rows


#    
#S0 = cv.imread('Test_image.JPG')
##S0 = S0[:,:,0]
#cv.imshow('S0',S0)
#
#
#
#S1 = cv.pyrDown(S0)
#cv.imshow('S1',S1)
#
#S2 = cv.pyrUp(S1,dstsize=(S0.shape[1],S0.shape[0]))
#cv.imshow('S2',S2)
#
##S3 = cv.subtract(S0,S2)
##S0 = np.int32(S0)
##S2 = np.int32(S2)
##S3 = S0-S2
#S3 = np.subtract(S0,S2)
#
#cv.imshow('S3',S3)
#
#S4 = S2*S3
#cv.imshow('S4',S4)
#
##size = (S.shape[1],S0.shape[0])
##S4 = np.add(S0, cv.pyrUp(S2,dstsize=size))
#cv.waitKey(0)
#cv.destroyAllWindows()
