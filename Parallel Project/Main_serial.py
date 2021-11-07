from __future__ import (division, absolute_import, print_function, unicode_literals)
import cv2 as cv
import numpy as np
import scipy.ndimage
#from scipy import ndimage
import copy
#import os
#from multiprocessing import Pool
#import psutil

img = cv.imread('Capture.jpg')

cv.imshow('Original',img)

###############################    Functions     ############################
###************  Input Image   ***************
def white_balance(img):
    result = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2RGB)
    return result

def adap_hist_equal(lab):
	lab_planes = cv.split(lab)
	clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
	lab_planes[0] = clahe.apply(lab_planes[0])
	lab2 = cv.merge(lab_planes)
	img2 = cv.cvtColor(lab2, cv.COLOR_LAB2RGB)
	return img2,lab2

#**************    Weights    ****************
def laplacian_wt(R1):
	h=np.array([[0.16667,0.66667,0.16667],[0.66667,-3.33333,0.66667],[0.16667,0.66667,0.16667]])
	return abs(scipy.ndimage.convolve(R1, h, mode='nearest')) 
   #abs(imfilter(R1, fspecial('Laplacian'), 'replicate', 'conv'))

def local_contrast(R1):
	h= np.array([[0.0625, 0.25, 0.375, 0.25, 0.0625]])
	WC1 = scipy.ndimage.convolve(R1, np.dot(h.transpose(),h), mode='nearest')
    #filter weights array has incorrect shape
	WC1[np.where(WC1 > (np.pi/2.75))] = np.pi/2.75
	return np.power((R1 - WC1),2)

def saliency_detection(img):
	#Read image and blur it with a 3x3 or 5x5 Gaussian filter
	blur = cv.GaussianBlur(img,(5,5),0)
	lab = cv.cvtColor(blur, cv.COLOR_RGB2LAB)#OpenCV and skimage use D65 (which is a standard for srgb).
	l = (lab[:,:,0])##double has to be implemented later on
	lm = np.mean(np.mean(l))
	a = (lab[:,:,1])
	am = np.mean(np.mean(a))
	b = (lab[:,:,2])
	bm = np.mean(np.mean(b))  # not working
	sm = np.power((l-lm),2) + np.power((a-am),2) + np.power((b-bm),2) #need to calculate b
	return sm
	
def exposedness_wt(R1):
	sigma = 0.25;
	aver = 0.5;
	return np.exp(-np.power((R1 - aver),2) / np.power(2*sigma,2) )

#***************     Fusion   ***************

def gaussian_pyramid(W,level):
	G = W.copy()
	Weight = [G]
	for i in range(level):
		G = cv.pyrDown(G)
		Weight.append(G)
	return Weight

def laplacian_pyramid(img,level,dtype=np.int16):
    img = dtype(img)
    lp = []
    for i in range(level-1):
        next_img = cv.pyrDown(img)
        #img1 = cv.pyrUp(next_img, dstsize=os.path.getsize(img))
        img1 = cv.pyrUp(next_img, dstsize=(img.shape[1],img.shape[0]))
        lp.append(img-img1)
        img = next_img
    lp.append(img)
    return np.array(lp)

def pyramid_reconstruct(pyramid):
	level=len(pyramid)
	for i in range(level-1,1,-1):
		a,b=pyramid[i-1].shape
		#print (a,b)
		pyramid[i-1]=pyramid[i-1]+ scipy.misc.imresize(pyramid[i], [a,b])
		return pyramid[0]
    
############################     Actual Implementation    ###########################
#########   1st Step    ############
        
img1 = white_balance(img)

lab1 = cv.cvtColor(img1, cv.COLOR_RGB2LAB)
lab = copy.deepcopy(lab1)
img2,lab2 = adap_hist_equal(lab)

cv.imshow('img1',img1)
cv.imshow('img2',img2)

cv.imshow('lab1',lab1)
cv.imshow('lab2',lab2)

#########    2nd Step     ##############

R1 = (lab1[:, :, 0]) / 255
R2 = (lab2[:, :, 0]) / 255

cv.imshow('R1',R1)
cv.imshow('R2',R2)

##calculate laplacian contrast weight
WL1 = laplacian_wt(R1)
WL2 = laplacian_wt(R2)

cv.imshow('WL1',WL1)
cv.imshow('WL2',WL2)

##calculate Local contrast weight
WC1 = local_contrast(R1)
WC2 = local_contrast(R2)

cv.imshow('WC1',WC1)
cv.imshow('WC2',WC2)

##calculate the saliency weight
WS1 = saliency_detection(img1)
WS2 = saliency_detection(img2)

cv.imshow('WS1',WS1)
cv.imshow('WS2',WS2)

## calculate the exposedness weight
WE1 = exposedness_wt(R1)
WE2 = exposedness_wt(R2)

cv.imshow('WE1',WE1)
cv.imshow('WE2',WE2)

#calculation of normalised weights
W1 = np.divide((WL1 + WC1 + WS1 + WE1),(WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2))
W2 = np.divide((WL2 + WC2 + WS2 + WE2),(WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2))

cv.imshow('W1',W1)
cv.imshow('W2',W2)

############    3rd Step      #############

#calculate the gaussian_pyramid
level = 5
Weight1 = gaussian_pyramid(W1, level)
Weight2 = gaussian_pyramid(W2, level)

#calculate the laplacian pyramid
# input1
R1 = laplacian_pyramid(img1[:, :, 0], level)
G1 = laplacian_pyramid(img1[:, :, 1], level)
B1 = laplacian_pyramid(img1[:, :, 2], level)
# input2
R2 = laplacian_pyramid(img2[:, :, 0], level)
G2 = laplacian_pyramid(img2[:, :, 1], level)
B2 = laplacian_pyramid(img2[:, :, 2], level)

# fusion 
R_r=[]
R_g=[]
R_b=[]
for i in range(0, level):
   R_r.append(np.multiply(Weight1[i],R1[i]) + np.multiply(Weight2[i],R2[i]))
   R_g.append(np.multiply(Weight1[i],G1[i]) + np.multiply(Weight2[i],G2[i]))
   R_b.append(np.multiply(Weight1[i],B1[i]) + np.multiply(Weight2[i],B2[i]))

# reconstruct & output
R = pyramid_reconstruct(R_r)
G = pyramid_reconstruct(R_g)
B = pyramid_reconstruct(R_b)

cv.imshow('R', R)
cv.imshow('G', G)
cv.imshow('B', B)

fusion = img
fusion[:,:,0] = R
fusion[:,:,1] = G
fusion[:,:,2] = B

cv.imshow('result', fusion)

cv.waitKey(0)
cv.destroyAllWindows()
