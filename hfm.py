#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2 as cv
import pygad
import numpy as np
import matplotlib.pyplot as plt
import skimage
import random


img = cv.imread("img/low.jpg", 0)
L = 256
maximum = 255
minimum = 0
hist = cv.calcHist([img],[0],None,[256],[0,256])
hs = (np.percentile(hist, 75) - np.percentile(hist, 25))/(maximum-minimum)
r = hist**(1/L)
g = np.prod(r, where = r > 0)  
a = np.sum(hist/L)
hfm =  g/a
print("hfm: {hfm}  hs:{hs}".format(hfm = hfm, hs = hs) )
