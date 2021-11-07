from __future__ import (division, absolute_import, print_function, unicode_literals)
import cv2 as cv
import numpy as np
import scipy
from scipy import ndimage
import copy
# Insert any filename with path
img = cv.imread('3.jpg')

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

def laplacian_wt(R1):
	h=np.array([[0.16667,0.66667,0.16667],[0.66667,-3.33333,0.66667],[0.16667,0.66667,0.16667]])
	return abs(scipy.ndimage.convolve(R1, h, mode='nearest')) #abs(imfilter(R1, fspecial('Laplacian'), 'replicate', 'conv'))

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
	
def local_contrast(R1):
	h= np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])
	WC1 = scipy.ndimage.convolve(R1, np.dot(h.conj().transpose(),h), mode='nearest') #filter weights array has incorrect shape
	WC1[np.where(WC1 > (np.pi/2.75))] = np.pi/2.75
	return np.power((R1 - WC1),2)

def gaussian_pyramid(W,level):
	G = W.copy()
	Weight = [G]
	for i in xrange(level):
		G = cv.pyrDown(G)
		Weight.append(G)
	return Weight

def laplacian_pyramid(img, level):
	gp=gaussian_pyramid(img,level)
	lp = [gp[4]]
	for i in xrange(5,0,-1):
		GE = cv.pyrUp(gp[i])
		L = cv.subtract(gp[i-1],GE)
		lp.append(L)
	return lp

def pyramid_reconstruct(pyramid):
	level=pyramid.size()
	for i in range(level,0,-1):
		a=pyramid.shape[i-1]
		pyramid[i-1]=pyramid[i-1]+ scipy.misc.imresize(pyramid[i], a)
		return pyramid[0]


#white balance to create 1st input
img1=white_balance(img)


final = np.hstack((img, img1))
cv.imshow('Temple', final)
cv.imwrite('result.jpg', final)

#RGB to Lab
lab1 = cv.cvtColor(img1, cv.COLOR_RGB2LAB)
lab=copy.deepcopy(lab1)
#contrast limited adaptive histogram equilisation to create 2nd input
img2,lab2=adap_hist_equal(lab)


cv.imshow('org', img)
cv.imshow('hey', img2)
cv.imshow('Temple', img2)

# Creating filters for both the inputs


R1 = (lab1[:, :, 0]) / 255
#calculate laplacian contrast weight
WL1 = laplacian_wt(R1)
#calculate Local contrast weight
WC1 = local_contrast(R1)
#calculate the saliency weight
WS1 = saliency_detection(img1)
# calculate the exposedness weight
WE1 = exposedness_wt(R1)


R2 = (lab2[:, :, 0]) / 255
#calculate laplacian contrast weight
WL2 = laplacian_wt(R2)# scipy.ndimage.laplace(R1,WL1,mode='nearest')
#calculate Local contrast weight
WC2 = local_contrast(R2)
#calculate the saliency weight
WS2 = saliency_detection(img2)
# calculate the exposedness weight
WE2 = exposedness_wt(R2)


#calculation of normalised weights

W1 = np.divide((WL1 + WC1 + WS1 + WE1),(WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2))
W2 = np.divide((WL2 + WC2 + WS2 + WE2),(WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2))

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
for i in range(0, 3):
   R_r[i] = np.multiply(Weight1[i],R1[i]) + np.multiply(Weight2[i],R2[i])
   R_g[i] = np.multiply(Weight1[i],G1[i]) + np.multiply(Weight2[i],G2[i])
   R_b[i] = np.multiply(Weight1[i],B1[i]) + np.multiply(Weight2[i],B2[i])


# reconstruct & output
R = pyramid_reconstruct(R_r)
G = pyramid_reconstruct(R_g)
B = pyramid_reconstruct(R_b)

fusion = np.concatenate(np.uint8(R), np.uint8(G), np.uint8(B),axis=2)

cv.imshow('result', fusion)


cv.waitKey(0)
cv.destroyAllWindows()


#https://isaacchanghau.github.io/post/underwater_image_fusion/