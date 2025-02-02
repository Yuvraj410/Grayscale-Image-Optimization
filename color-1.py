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
from skimage.util.shape import view_as_windows


n = 3
filter_size = (n, n)
w = filter_size[0]//2
x = (1 - filter_size[0]%2) - (filter_size[0]//2)
y = filter_size[1]//2
z = (1 - filter_size[1]%2) - (filter_size[1]//2)



bgrImage = cv.imread('img/imggg.png')
hsvImage = cv.cvtColor(bgrImage, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsvImage)
image = np.copy(v)


img1 = np.pad(image, ((1, 1), (1, 1)), mode='constant')

# Set window size
win_size = (3, 3)  # (height, width)

# Extract patches using sliding window
patches = view_as_windows(img1, win_size)

# Calculate local arithmetic mean of each patch
arithmetic_mean = np.mean(patches, axis=(2, 3))

# Calculate local geometric mean of each patch
nonzero_patches = patches.copy()
nonzero_patches[nonzero_patches == 0] = 1  # Avoid taking the logarithm of zero
geometric_mean = np.exp(np.mean(np.log(nonzero_patches), axis=(2, 3)))

gain = np.zeros_like(image)
gain = np.divide(geometric_mean, arithmetic_mean)


def imageTransform(img, a, c):
    imgH, imgW =  np.shape(img)
    temp = np.zeros(np.shape(img), dtype="float32")
    global arithmetic_mean
    for i in range(1, imgH-1):
        for j in range(1, imgW-1):
            m = arithmetic_mean[i-1, j-1]
            temp[i][j] = gain[i][j]*(img[i][j]-c*m) + m**(a)
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
    hist = cv.calcHist([im],[0],None,[256],[0,256])
    r = np.power(hist, 1/L)
    g = np.prod(r, where = r > 0)  
    a = np.sum(hist)/L
    hfm =  g/a
    hs = histogram_spread(hist)
    return hfm, hs


def preprocess(im):
    global s, v, r
    exposure = calculate_exposure(v)
    print("exposure={exp}".format(exp = exposure))
    p = 1-0.5*exposure
    print("p={p}".format(p = p))
    s = np.power(s, p)
    s = s.astype('uint8')
    r = 1 if exposure < 0.5 else -1

    
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
# c__ = np.linspace(1, 5, 500, endpoint=True)

ga_instance = pygad.GA(num_generations=30,
                        num_parents_mating=2, 
                        fitness_func=fitness_function,
                        initial_population=[
                            [0.1, 0.2],
                            [.3, .1],
                            [.5, .4],
                            [.7, .4],
                            [.1, .5]],
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
                        # gene_space=[a__, c__],
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



