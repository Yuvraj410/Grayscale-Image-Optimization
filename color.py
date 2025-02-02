#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import pygad
import numpy as np
import matplotlib.pyplot as plt
import skimage
from datetime import datetime
from scipy.ndimage import generic_filter
from sklearn.preprocessing import normalize

n = 3
filter_size = (n, n)
w = filter_size[0]//2
x = (1 - filter_size[0]%2) - (filter_size[0]//2)
y = filter_size[1]//2
z = (1 - filter_size[1]%2) - (filter_size[1]//2)



bgrImage = cv.imread('img/Lenna-loww.png')
hsvImage = cv.cvtColor(bgrImage, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsvImage)
image = np.copy(v)


def local_am(img):
    # Convert the image to a floating-point array
    img = np.array(img, dtype=np.float32)
    
    # Pad the image with zeros to preserve the size of the output
    img = np.pad(img, ((1, 1), (1, 1)), mode='constant')
    
    # Create a 3x3 filter with all elements set to 1/9
    filter = np.ones((3, 3), dtype=np.float32) / 9
    
    # Calculate the mean image using a convolution
    mean = cv.filter2D(img, -1, filter, borderType=cv.BORDER_CONSTANT)
    
    # Trim the padding to obtain the original size of the image
    mean = mean[1:-1, 1:-1]
    
    # Convert the result back to 8-bit unsigned integers
    mean = np.uint8(mean)
    return mean

def local_gm(image, window_size):
    image_padded = np.pad(image, window_size, mode='reflect')
    result = np.zeros(image.shape)
    for i in range(window_size, image.shape[0] + window_size):
        for j in range(window_size, image.shape[1] + window_size):
            window = image_padded[i - window_size:i + window_size + 1, j - window_size:j + window_size + 1]
            result[i - window_size, j - window_size] = np.power(np.prod(window), 1 / (2 * window_size + 1) ** 2)
    return result


arith_mean_arr = local_am(image)

geom_mean_arr = local_gm(image, 3)

   
def calculateGain(img, i, j, n):
    imgH, imgW =  np.shape(img)
    return geom_mean_arr[i, j]/arith_mean_arr[i, j]


def imageTransform(img, a, c):
    imgH, imgW =  np.shape(img)
    temp = np.zeros(np.shape(img), dtype="float32")
    global arith_mean_arr, geom_mean_arr
    for i in range(1, imgH-1):
        for j in range(1, imgW-1):
            m = arith_mean_arr[i-1, j-1]
            gain = calculateGain(img, i-1, j-1, 3)
            temp[i][j] = gain*(img[i][j]-c*m) + m**(a)
    # print("max val in image : {m}".format(m = np.amax(temp)))
    max_val = np.amax(temp)
    # if there is very few in max val, values are shifting towards black
    # if max_val>255:
    #     med = np.median(temp)
    #     diff = med - 128
    #     temp = temp - diff

    temp = np.clip(temp, 0, 255)
    temp = temp.astype('uint8')
    return temp

def calculate_exposure(im):
    hist = cv.calcHist([im],[0],None,[256],[0,256]).ravel()
    r, c =  np.shape(im)
    d = r*c*255
    bins = np.arange(256)
    return hist.dot(bins)/d


def calculate_HFM_HS(im):
    L = 256
    hist = cv.calcHist([im],[0],None,[256],[0,256])
    hs = (np.percentile(hist, 75) - np.percentile(hist, 25))/255
    r = np.power(hist, 1/L)
    g = np.prod(r, where = r > 0)  
    a = np.sum(hist)/L
    hfm =  g/a
    return hfm, hs


def preprocess(im):
    global s, v, r
    exposure = calculate_exposure(v)
    print("exposure={exp}".format(exp = exposure))
    p = 1-0.5*exposure
    print("p={p}".format(p = p))
    s = np.power(s, p)
    s = s.astype('uint8')
    r = 1 if exposure < 0.5 else 1

    
r = 1

preprocess(hsvImage)

def fitness_function(solution, solution_idx):
    a,c = solution
    global image, r
    temp = imageTransform(image, a, c)
    hfm, hs = calculate_HFM_HS(temp)
    entropy = skimage.measure.shannon_entropy(temp)
    exposure = calculate_exposure(temp)
    fitness =  entropy*hfm*hs*(exposure**r)
    # fitness = hfm * hs * entropy
    return fitness


last_fitness = 0
count = 0
def on_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))    
    print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    
    
a__ = np.linspace(0.0, 1.5, 500, endpoint=True)
c__ = np.linspace(0.1, 5, 500, endpoint=True)

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

v = imageTransform(v, a, c)

hsv_image = cv.merge([h, s, v])
out = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)


cv.imwrite("results/resColor_{now}.png".format(now=datetime.now()), out)


fig = plt.figure(figsize=(6, 6))

fig.add_subplot(221)
plt.imshow(cv.cvtColor(bgrImage, cv.COLOR_BGR2RGB))

fig.add_subplot(222)
plt.imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB))

fig.add_subplot(223)
plt.hist(image.ravel(), 256,[0,256])
plt.show()

fig.add_subplot(224)
plt.hist(v.ravel(), 256,[0,256]); 
plt.show()
plt.savefig('results/res_{now}.png'.format(now = datetime.now()))


cv.waitKey(0)
cv.destroyAllWindows()


