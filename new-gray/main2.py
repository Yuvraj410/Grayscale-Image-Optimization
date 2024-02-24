#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:28:38 2023

@author: daniyal
"""

import cv2 as cv
import pygad
import numpy as np
import matplotlib.pyplot as plt
import skimage


img = cv.imread("../img/balloons_lowcontrast_lowbrightness.png", 0)

def gm(img, i, j, n):
    temp = np.copy(img[i-n//2:i+n//2+1, j-n//2:j+n//2+1])
    temp.astype("float32")
    temp = np.power(temp, 1/(n*n))
    return np.prod(temp, where = temp>0)


def am(img, i, j, n):
    temp =  np.copy(img[i-n//2:i+n//2+1, j-n//2:j+n//2+1])
    temp.astype("float32")
    res = np.sum(temp)/(n*n)
    return  res if res>0 else 1
   
def calculateGain(img,i, j, n):
    imgH, imgW =  np.shape(img)
    return gm(img, i, j, n)/am(img, i, j, n)

def imageTransform(img, a, c):
    imgH, imgW =  np.shape(img)
    temp = np.zeros(np.shape(img), dtype="float32")
    for i in range(imgH):
        for j in range(imgW):
            m = am(img, i, j, 3)
            gain = calculateGain(img, i, j, 3)
            temp[i][j] = gain*(img[i][j]-c*m) + m**a
    temp = np.clip(temp, 0, 255)
    temp = temp.astype('uint8')
    return temp

def calculate_exposure(img):
    hist = cv.calcHist([img],[0],None,[256],[0,256]).reshape((256,))
    r, c =  np.shape(img)
    d = r*c*255
    bins = np.arange(256)
    bins = bins
    return hist.dot(bins)/d

# Function to calculate histogram spread
def histogram_spread(histogram):
    cdf = np.cumsum(histogram)
    ncdf = cdf / np.sum(histogram)
    bin_75 = np.argwhere(ncdf >= 0.75)[0][0]
    bin_25 = np.argwhere(ncdf >= 0.25)[0][0]

    HS = (bin_75 - bin_25)/255
    return HS


def calculate_HFM_HS(im):
    L = 256
    histogram = cv.calcHist([im],[0],None,[256],[0,256])
    hs = histogram_spread(histogram)
    r = np.power(histogram, 1/L)
    g = np.prod(r, where = r > 0)  
    a = np.sum(histogram)/L
    hfm =  g/a
    return hfm, hs

# Function to calculate histogram spread
def histogram_spread(histogram):
    cdf = np.cumsum(histogram)
    ncdf = cdf / np.sum(histogram)
    bin_75 = np.argwhere(ncdf >= 0.75)[0][0]
    bin_25 = np.argwhere(ncdf >= 0.25)[0][0]
    HS = (bin_75 - bin_25)/255
    return HS



orig_exposure = calculate_exposure(img)
p = 1 if orig_exposure < 0.5 else -1


def fitness_function(solution, solution_idx):
    a,c = solution
    global img, p
    temp = imageTransform(np.copy(img), a, c)
    hfm, hs = calculate_HFM_HS(temp)
    entropy = skimage.measure.shannon_entropy(temp)
    # exposure = calculate_exposure(temp)
    # fitness =  entropy*hfm*hs*(exposure**p)
    fitness = hfm * hs* entropy
    return fitness


last_fitness = 0
count = 0
def on_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))    
    print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    # if ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness==0:
    #     return "stop"

a__ = np.linspace(0.0, 1.5, 100, endpoint=True)
ga_instance = pygad.GA(num_generations=5,
                        num_parents_mating=2,
                        fitness_func=fitness_function,
                        init_range_high=1,
                        init_range_low=0,
                        sol_per_pop=5, 
                        num_genes=2,
                        mutation_type="random",
                        mutation_by_replacement=True,
                        mutation_num_genes=1,
                        mutation_percent_genes=0.01,
                        random_mutation_max_val=1,
                        random_mutation_min_val=0,
                        gene_space=[a__, None],
                        on_generation=on_generation
                        )


ga_instance.run()

ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

print("optimal a:{a} and c:{c}".format(a=solution[0], c= solution[1]))


a, c = solution

final_image = imageTransform(img, a, c)

cv.imwrite("res_nov17.png", final_image)

fig = plt.figure(figsize=(6, 6))

fig.add_subplot(221)
plt.imshow(img, cmap='gray')


fig.add_subplot(222)
plt.imshow(final_image, cmap='gray')

fig.add_subplot(223)
plt.hist(img.ravel(), 256,[0,256])
plt.show()

fig.add_subplot(224)
plt.hist(final_image.ravel(), 256,[0,256]); 
plt.show()


cv.waitKey(0)
cv.destroyAllWindows()

