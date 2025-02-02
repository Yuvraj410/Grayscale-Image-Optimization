#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import pygad
import numpy as np
import matplotlib.pyplot as plt
import skimage
from datetime import datetime
from scipy.ndimage import generic_filter

n = 3
filter_size = (n, n)
w = filter_size[0]//2
x = (1 - filter_size[0]%2) - (filter_size[0]//2)
y = filter_size[1]//2
z = (1 - filter_size[1]%2) - (filter_size[1]//2)



image = cv.imread("img/x31_f18.png", 0)
image1 = np.copy(image)

def Func(a):
    return np.prod( np.power(a, 1/9))


def am(img):
    global arith_mean_arr, filter_size, w, x, y, z
    arith_mean_arr = generic_filter(img, np.average, filter_size)[w:x or None, y:z or None]


def gm(img):
    global filter_size, geom_mean_arr, w, x, y, z
    geom_mean_arr = generic_filter(image, Func, size=filter_size)[w:x or None, y:z or None]


arith_mean_arr = generic_filter(image, np.average, filter_size)[w:x or None, y:z or None]
geom_mean_arr = generic_filter(image, Func, size=filter_size)[w:x or None, y:z or None]

   
def calculateGain(img,i, j, n):
    imgH, imgW =  np.shape(img)
    return geom_mean_arr[i, j]/arith_mean_arr[i, j]

am(image)
gm(image)
 
def imageTransform(img, a, c):
    imgH, imgW =  np.shape(img)
    temp = np.zeros(np.shape(img), dtype="float32")
    global arith_mean_arr, geom_mean_arr
    # am(img)
    # gm(img)
    for i in range(1, imgH-1):
        for j in range(1, imgW-1):
            m = arith_mean_arr[i-1, j-1]
            gain = calculateGain(img, i-1, j-1, 3)
            temp[i][j] = gain*(img[i][j]-c*m) + m**a
    # print("max val in image : {m}".format(m = np.amax(temp)))
    max_val = np.amax(temp)
    if max_val>255:
        med = np.median(temp)
        diff = med - 128
        temp = temp - diff
        
    temp = np.clip(temp, 0, 255)
    temp = temp.astype('uint8')
    return temp

# def calculate_exposure(img):
#     hist = cv.calcHist([img],[0],None,[256],[0,256])
#     r, c =  np.shape(img)
#     d = r*c*255
#     bins = np.arange(256)
#     bins = bins
#     return hist.dot(bins)/d


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



# orig_exposure = calculate_exposure(image)
# p = 1 if orig_exposure < 0.5 else -1


def fitness_function(solution, solution_idx):
    a,c = solution
    global image, p
    temp = imageTransform(np.copy(image), a, c)
    hfm, hs = calculate_HFM_HS(temp)
    entropy = skimage.measure.shannon_entropy(temp)
    # exposure = calculate_exposure(temp)
    # fitness =  entropy*hfm*hs*(exposure**p)
    fitness = hfm * hs * entropy
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

a__ = np.linspace(0.0, 1.5, 500, endpoint=True)
c__ = np.linspace(0, 5, 500, endpoint=True)

ga_instance = pygad.GA(num_generations=30,
                        num_parents_mating=2, 
                        fitness_func=fitness_function,
                        # init_range_high=1,
                        # init_range_low=0.0,
                        sol_per_pop=5, 
                        num_genes=2,
                        mutation_type="random",
                        mutation_by_replacement=True,
                        mutation_num_genes=1,
                        mutation_percent_genes=0.01,
                        # random_mutation_max_val=1,
                        # random_mutation_min_val=0,
                        gene_space=[a__, c__],
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

final_image = imageTransform(image, a, c)

cv.imwrite("res_{now}.png".format(now=datetime.now()), final_image)

fig = plt.figure(figsize=(6, 6))

fig.add_subplot(221)
plt.imshow(image, cmap='gray')


fig.add_subplot(222)
plt.imshow(final_image, cmap='gray')

fig.add_subplot(223)
plt.hist(image.ravel(), 256,[0,256])
plt.show()

fig.add_subplot(224)
plt.hist(final_image.ravel(), 256,[0,256]); 
plt.show()


cv.waitKey(0)
cv.destroyAllWindows()


