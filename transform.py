#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2 as cv
import pygad
import numpy as np
import matplotlib.pyplot as plt
import skimage


img = cv.imread("img/balloons_lowcontrast_lowbrightness.png", 0)

def gm(img, i, j, n):
    temp = img[i-n//2:i+n//2+1, j-n//2:j+n//2+1]
    temp.astype("float32")
    temp = np.power(temp, 1/(n*n))
    return np.prod(temp, where = temp>0)


def am(img, i, j, n):
    temp = img[i-n//2:i+n//2+1, j-n//2:j+n//2+1]
    temp.astype("float32")
    res = np.sum(temp)/(n*n)
    return  res if res>0 else 1
   
def calculateGain(img,i, j, n):
    imgH, imgW =  np.shape(img)
    return gm(img, i, j, n)/am(img, i, j, n)


def imageTransform(img, a, c):
    imgH, imgW =  np.shape(img)
    temp = np.zeros(np.shape(img))
    for i in range(imgH):
        for j in range(imgW):
            m = am(img, i, j, 3)
            gain = calculateGain(img, i, j, 3)
            temp[i][j] = gain*(img[i][j]-c*m) + m**a
    temp = np.clip(temp, 0, 255)
    temp = temp.astype('uint8')
    return temp



def calculate_HFM_HS(im):
    L = 256
    maximum = 255
    minimum = 0
    hist = cv.calcHist([im],[0],None,[256],[0,256])
    hs = (np.percentile(hist, 75) - np.percentile(hist, 25))/255
    r = np.power(hist, 1/L)
    g = np.prod(r, where = r > 0)  
    a = np.sum(hist)/L
    hfm =  g/a
    return hfm, hs

# fig = plt.figure(figsize=(8, 8))
# fig.tight_layout(pad=10.0)
# columns = 8
# rows = 4

# ax= []


a__balloons = np.linspace(0.2, 1.5, 10)
c__balloons = np.linspace(0, 0.99, 10)
r = np.shape(a__balloons)[0]

c = np.shape(c__balloons)[0]

fitnesses_balloons = np.empty((r, c))
hfms_balloons= np.copy(fitnesses_balloons)
hss_balloons = np.copy(fitnesses_balloons)
entropies_balloons = np.copy(fitnesses_balloons)
k = 1
# for i, a in enumerate(a__balloons):
#     for j, c in enumerate(c__balloons):
#         print(k)
#         final_image = imageTransform(img, a, c)
#         hfm, hs = calculate_HFM_HS(final_image)
#         entropy = skimage.measure.shannon_entropy(final_image)
#         fitness = hfm*hs*entropy
#         hfms_balloons[i][j]=hfm
#         hss_balloons[i][j]=hs
#         fitnesses_balloons[i][j]= fitness
#         fitnesses_balloons[i][j] = entropy
#         # print("a={:.2f} c={:.2f} hfm={:.2f} hs = {:.2f} entropy={:.2f} fitness={fitness}".format(a, c, hfm, hs, entropy, fitness=fitness))
#         # cv.imwrite("generated/{a}_{c}.png".format(a=a, c=c), final_image)
#         # ax.append( fig.add_subplot(rows, columns, k) )
#         # ax[-1].set_title("a = {:.2f}   c = {:.2f}".format(a, c))
#         # plt.imshow(final_image, cmap='gray')
#         # cv.imwrite("generated/balloons/balloons1_{a}_{c}.png".format(a=a,c=c), final_image)

        
#         k+=1 

final_image = imageTransform(img, 1.07, 0.99)

# fig = plt.figure(figsize=(4, 4))
# fig.add_subplot(311)
# res = np.hstack((img, final_image))

plt.imshow(final_image, cmap='gray')


# fig.add_subplot(312)
# plt.hist(img.ravel(), 256,[0,256])
# plt.show()

# fig.add_subplot(313)
# plt.hist(final_image.ravel(), 256,[0,256]); 
# plt.show()

        
# fig.tight_layout()
# plt.show()
        
