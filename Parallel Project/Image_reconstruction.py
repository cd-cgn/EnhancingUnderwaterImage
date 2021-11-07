# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:09:04 2019

@author: SHIKHAR GUPTA
"""
import cv2 as cv
import numpy as np

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

img = cv.imread('buddha.JPG')
img = cv.imread('messi.JPG')
cv.imshow('img',img)

G = gaussian_pyramid(img,5)
L = laplacian_pyramid(img,5)

cv.imshow('G[0]',G[0])
cv.imshow('G[1]',G[1])
cv.imshow('L[0]',L[0])
cv.imshow('L[1]',L[1])

K = cv.pyrUp(G[1],dstsize=(L[0].shape[1],L[0].shape[0]))
cv.imshow('K',K)
X = np.add(K,L[0])
cv.imshow('X',X)


cv.waitKey(0)
cv.destroyAllWindows()