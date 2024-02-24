#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:51:02 2022

@author: daniyal
"""
import cv2 as cv
import pygad
import numpy as np
import matplotlib.pyplot as plt
import skimage
import random
from datetime import date
from matplotlib.colors import hsv_to_rgb


img = cv.imread('img/Lenna.png')
hsvImage = cv.cvtColor(img, cv.COLOR_BGR2HSV)
final_img = np.copy(hsvImage)

def gm(i, j, n):
    global hsvImage
    temp = hsvImage[i-n//2:i+n//2, j-n//2:j+n//2, 2]
    temp = temp ** (1/(n*n))
    g = np.prod(temp, where=temp>0)
    return g
    

def am(i, j, n):
    global hsvImage
    temp = np.copy(hsvImage[i-n//2:i+n//2, j-n//2:j+n//2, 2])
    temp.astype('float32')
    return np.sum(temp)/(n*n)
   
def calculateGain(i, j, n):
    global hsvImage
    imgH, imgW =  np.shape(hsvImage[:, :, 2])
    gg=gm(i, j, n)
    amm = am(i, j, n)
    if amm>0:
        return gg/amm
    return 0

def imageTransform(img, a, c):
    imgH, imgW =  np.shape(img)
    temp = np.zeros( np.shape(img))
    for i in range(imgH):
        for j in range(imgW):
            mean = am(i, j, 3)
            gain = calculateGain(i, j, 3)
            temp[i][j] = gain*(img[i][j]-c*mean)+mean**a
    temp = np.clip(temp, 0, 255)
    temp = temp.astype('uint8')
    return temp


def calculate_exposure(img):
    hist = cv.calcHist([img],[0],None,[256],[0,256]).reshape((256,))
    r, c =  np.shape(img)
    d = r*c*255
    bins = np.arange(256)
    bins = bins/d
    return hist.dot(bins)


def calculate_HFM_HS(img):
    L = 256
    maximum = 255
    minimum = 0
    hist = cv.calcHist([img],[0],None,[256],[0,256])
    hs = (np.percentile(hist, 75) - np.percentile(hist, 25))/255
    r = hist**(1/L)
    g = np.prod(r, where = r > 0)  
    a = np.sum(hist/L)
    hfm =  g/a
    return hfm, hs



orig_exposure = calculate_exposure(hsvImage[:, :, 2])
p = 1 if orig_exposure < 0.5 else -1

final_img[:, :, 1] = np.copy(hsvImage[:, :, 1] ** (1-0.5*orig_exposure))

def fitness_function(solution, solution_idx):
    a,c = solution
    global hsvImage, p
    temp = imageTransform(hsvImage[:,:,2], a, c)
    hfm, hs = calculate_HFM_HS(temp)
    entropy = skimage.measure.shannon_entropy(temp)
    exposure = calculate_exposure(temp)
    fitness =  entropy*hfm*hs*(exposure**p)
    print("entropy: {:.2f}  hs: {:.2f} hfm: {:.2f}".format(entropy, hs, hfm))
    print("fitness: {fitness}".format(fitness=fitness))
    return fitness


last_fitness = 0
count = 0
def on_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))    
    print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    if ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness==0:
        return "stop"

ga_instance = pygad.GA(num_generations=3,
                        num_parents_mating=3,
                        fitness_func=fitness_function,
                        init_range_high=1.1,
                        init_range_low=0.1,
                        sol_per_pop=10, 
                        num_genes=2,
                        mutation_type="random",
                        mutation_by_replacement=True,
                        mutation_num_genes=1,
                        mutation_percent_genes=0.01,
                        on_generation=on_generation)


ga_instance.run()

ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

print("optimal a:{a} and c:{c}".format(a=solution[0], c= solution[1]))


a, c = solution

final_img[:, :, 2] = imageTransform(img[:, :, 2], a, c)

cv.imwrite("res_{date}.png".format(date = date.today()), final_img)

fig = plt.figure(figsize=(4, 4))

fig.add_subplot(221)
plt.imshow(img)


fig.add_subplot(222)
f = cv.cvtColor(final_img, cv.COLOR_HSV2BGR)
plt.imshow(f)

fig.add_subplot(223)
plt.hist(hsvImage[2].ravel(), 256,[0,256])
plt.show()

fig.add_subplot(224)
plt.hist(final_img[2].ravel(), 256,[0,256]); 
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
