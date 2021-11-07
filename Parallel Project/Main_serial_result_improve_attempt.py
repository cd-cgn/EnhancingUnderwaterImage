from __future__ import (division, absolute_import, print_function, unicode_literals)
import cv2 as cv
import numpy as np
import scipy.ndimage
#from scipy import ndimage
import copy
#import os
#from multiprocessing import Pool
#import psutil

img = cv.imread('buddha.jpg')

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
	gpA = [G]
	for i in range(level):
		G = cv.pyrDown(G)
		gpA.append(G)
	return gpA

def laplacian_pyramid(img,level):
    G = img.copy()
    gpA = [G]
    for i in range(level):
        G = cv.pyrDown(G)
        gpA.append(G)
    lpA = [gpA[4]]
    for i in range(level-1,0,-1):
        GE = cv.pyrUp(gpA[i], dstsize=(gpA[i-1].shape[1],gpA[i-1].shape[0]))
        L = np.subtract(gpA[i-1],GE)
        lpA.append(L)
    lpA.reverse()
    return lpA
    
def pyramid_reconstruct(pyr,level):
    for i in range(level-1,1,-1):
        size =(pyr[i-1].shape[1],pyr[i-1].shape[0])
        pyr[i-1] = np.add(pyr[i-1], cv.pyrUp(pyr[i],dstsize=size))
    return pyr[0]


    
############################     Actual Implementation    ###########################
#########   1st Step    ############
        
img1 = white_balance(img)

lab1 = cv.cvtColor(img1, cv.COLOR_RGB2LAB)
lab = copy.deepcopy(lab1)
img2,lab2 = adap_hist_equal(lab)

#cv.imshow('img1',img1)
#cv.imshow('img2',img2)

#cv.imshow('lab1',lab1)
#cv.imshow('lab2',lab2)

#########    2nd Step     ##############

R1 = (lab1[:, :,0]) / 255
R2 = (lab2[:, :,0]) / 255

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

############    3rd Step      #############
###########  Linear Fusion
final = img
final[:,:,0] = (np.multiply(W1,img1[:,:,0]) + np.multiply(W2,img2[:,:,0]))
final[:,:,1] = (np.multiply(W1,img1[:,:,1]) + np.multiply(W2,img2[:,:,1]))
final[:,:,2] = (np.multiply(W1,img1[:,:,2]) + np.multiply(W2,img2[:,:,2]))

cv.imshow('final',final)

level = 5
#calculate the gaussian_pyramid
WW1 = gaussian_pyramid(W1, level)
WW2 = gaussian_pyramid(W2, level)

#calculate the laplacian pyramid
# input1
R1 = laplacian_pyramid(img1[:, :, 0], level)
G1 = laplacian_pyramid(img1[:, :, 1], level)
B1 = laplacian_pyramid(img1[:, :, 2], level)
# input2
R2 = laplacian_pyramid(img2[:, :, 0], level)
G2 = laplacian_pyramid(img2[:, :, 1], level)
B2 = laplacian_pyramid(img2[:, :, 2], level)

cv.imshow('W1',W1)
cv.imshow('WW1[0]',WW1[0])
J = laplacian_pyramid(img1,level)
cv.imshow('img1',img1)
cv.imshow('J[0]',J[0])

k = np.zeros((img1.shape[0],img1.shape[1],3),np.uint8)
k[:,:,0] = R1[0]
k[:,:,1] = G1[0]
k[:,:,2] = B1[0]
cv.imshow('k', k)

cv.imshow('1',WW1[0])
cv.imshow('2',R1[0])
cv.imshow('3',WW2[0])
cv.imshow('4',R2[0])
cv.imshow('5',WW1[0]*R1[0]+WW2[0]*R2[0])

# fusion 
R_r=[]
R_g=[]
R_b=[]
for i in range(0, level):   
    R_r.append(WW1[i]*R1[i] + WW2[i]*R2[i])
    R_g.append(WW1[i]*G1[i] + WW2[i]*G2[i])
    R_b.append(WW1[i]*B1[i] + WW2[i]*B2[i])
    

cv.imshow('R_r[0]',R_r[0])

f = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
f[:,:,0] = R_r[0]
f[:,:,1] = R_g[0]
f[:,:,2] = R_b[0]
cv.imshow('f', f)


###########################################  here

R = pyramid_reconstruct(R_r,level)
G = pyramid_reconstruct(R_g,level)
B = pyramid_reconstruct(R_b,level)

cv.imshow('R', R)
cv.imshow('G', G)
cv.imshow('B', B)

fusion = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
fusion[:,:,0] = R
fusion[:,:,1] = G
fusion[:,:,2] = B

cv.imshow('result', fusion)

cv.waitKey(0)
cv.destroyAllWindows()
