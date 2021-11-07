from __future__ import (division, absolute_import, print_function, unicode_literals)
import cv2 as cv
#import numpy as np
import scipy.ndimage
import copy
import time

###############################    Functions     ############################
#  ************  Input Image   ***************

def white_balance(img):
    result = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    avg_a = sum(map(sum,result[:, :, 1]))/(len(result[:, 0, 0])+len(result[0, :, 0]))
    avg_b = sum(map(sum,result[:, :, 2]))/(len(result[:, 0, 0])+len(result[0, :, 0]))
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv.cvtColor(result, cv.COLOR_LAB2RGB)
    return result


def adap_hist_equal(lab):
    lab_planes = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab2 = cv.merge(lab_planes)
    img2 = cv.cvtColor(lab2, cv.COLOR_LAB2RGB)
    return img2, lab2


# **************    Weights    ****************
def laplacian_wt(R1): 
    ddepth = cv.CV_16S
    kernel_size = 3
    src = R1
    src = cv.GaussianBlur(src, (3, 3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dst = cv.Laplacian(src_gray, ddepth, kernel_size)
    abs_dst = cv.convertScaleAbs(dst)
    return abs_dst

def local_contrast(R1):
    
#def local_contrast(R1):
#    h = np.array([[0.0625, 0.25, 0.375, 0.25, 0.0625]])
#    WC1 = scipy.ndimage.convolve(R1, np.dot(h.transpose(), h), mode='nearest')
#    # filter weights array has incorrect shape
#    WC1[np.where(WC1 > (np.pi / 2.75))] = np.pi / 2.75
#    return np.power((R1 - WC1), 2)
#
#
def saliency_detection(img):
    blur = cv.GaussianBlur(img, (5, 5), 0)
    lab = cv.cvtColor(blur, cv.COLOR_RGB2LAB) 
    l = (lab[:, :, 0]) 
    lm = sum(map(sum,l)/(len(l[:])+len(l[0][:]))
    a = (lab[:, :, 1])
    am = sum(map(sum,a)/(len(a[:])+len(a[0][:]))
    b = (lab[:, :, 2])
    bm = sum(map(sum,b)/(len(b[:])+len(b[0][:]))
    sm = l
    for i in len(l[:]):
        for j in len(l[0][:]):
            sm[i][j] = pow(l[i][j]-lm,2) + pow(a[i][j]-am,2)+pow(b[i][j]-bm,2) 
    return sm

def exposedness_wt(R1):
    for i in len(R1[:]):
        for j in len(R1[0][:]):
            for k in len(R1[0][0][:]):
                R1[i][j][k] = math.exp(-pow(R1[i][j][k]-0.5,2)/pow(2*0.25, 2))
    return R1

# ***************     Fusion   ***************
def gaussian_pyramid(W,level=5):
	G = W.copy()
	gpA = [G]
	for i in range(level):
		G = cv.pyrDown(G)
		gpA.append(G)
	return gpA

def laplacian_pyramid(img,level=5):
    G = img.copy()
    gpA = [G]
    for i in range(level):
        G = cv.pyrDown(G)
        gpA.append(G)
    lpA = [gpA[4]]
    for i in range(level-1,0,-1):
        GE = cv.pyrUp(gpA[i], dstsize=(gpA[i-1].shape[1],gpA[i-1].shape[0]))
        L = cv.subtract(gpA[i-1],GE)
        lpA.append(L)
    lpA.reverse()
    return lpA
    
def pyramid_reconstruct(pyr,level=5):
    for i in range(level-1,1,-1):
        size =(pyr[i-1].shape[1],pyr[i-1].shape[0])
        pyr[i-1] = cv.add(pyr[i-1], cv.pyrUp(pyr[i],dstsize=size))
    return pyr[0]

############################     Actual Implementation    ###########################

def step1(img):
    img1 = white_balance(img)
    lab1 = cv.cvtColor(img1, cv.COLOR_RGB2LAB)
    lab = copy.deepcopy(lab1)
    img2, lab2 = adap_hist_equal(lab)

    R1 = (lab1[:, :, 0]) / 255
    R2 = (lab2[:, :, 0]) / 255

    cv.imshow('img1', img1)
    cv.imshow('img2', img2)

    cv.imshow('lab1', lab1)
    cv.imshow('lab2', lab2)

    cv.imshow('R1', R1)
    cv.imshow('R2', R2)

    return [img1, img2, R1, R2]

def step2(img1, img2, R1, R2):
    ##calculate laplacian contrast weight
    WL1 = laplacian_wt(R1)
    WL2 = laplacian_wt(R2)
    
    ##calculate Local contrast weight
    WC1 = local_contrast(R1)
    WC2 = local_contrast(R2)
    
    ##calculate the saliency weight
    WS1 = saliency_detection(img1)
    WS2 = saliency_detection(img2)
    
    ## calculate the exposedness weight
    WE1 = exposedness_wt(R1)
    WE2 = exposedness_wt(R2)
    
    W1 = (WL1 + WC1 + WS1 + WE1)/(WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2)
    W2 = (WL2 + WC2 + WS2 + WE2)/(WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2)

    cv.imshow('W1', W1)
    cv.imshow('W2', W2)

    return W1, W2

def step3(W1, W2, img1, img2):
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
        R_r.append(Weight1[i]*R1[i] + Weight2[i]*R2[i])
        R_g.append(Weight1[i]*G1[i] + Weight2[i]*G2[i])
        R_b.append(Weight1[i]*B1[i] + Weight2[i]*B2[i])
    
    # reconstruct & output
    R = pyramid_reconstruct(R_r)
    G = pyramid_reconstruct(R_g)
    B = pyramid_reconstruct(R_b)

    cv.imshow('R', R)
    cv.imshow('G', G)
    cv.imshow('B', B)

    fusion = img

    fusion[:, :, 0] = R
    fusion[:, :, 1] = G
    fusion[:, :, 2] = B

    return fusion

if __name__ == '__main__':
    start = time.time()
    
    img = cv.imread('shark.jpg')
    cv.imshow('original', img)
    
    [img1, img2, R1, R2] = step1(img)  ##### step1 
    
    W1, W2 = step2(img1, img2, R1, R2) ##### step2
    
    result = step3(W1, W2, img1, img2) ##### step3 

    cv.imshow('Result', result)
    
    finish = time.time()
    print(finish - start)

    cv.waitKey(0)
    cv.destroyAllWindows()
