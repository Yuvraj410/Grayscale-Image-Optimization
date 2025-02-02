# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2 as cv
import pygad
import numpy as np
import matplotlib.pyplot as plt
import skimage
from datetime import datetime
from skimage.util.shape import view_as_windows


image = cv.imread("/home/daniyal/image-processing/img/imgg.png", 0)



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
    temp = np.zeros_like(img, dtype='float32')
    global arithmetic_mean, geometric_mean, gain
    for i in range(0, imgH):
        for j in range(0, imgW):
            m = arithmetic_mean[i, j]
            temp[i][j] = gain[i][j]*(img[i][j]-c*m) + m**a
        
    temp = np.clip(temp, 0, 255)
    temp = temp.astype('uint8')
    return temp


def calculate_exposure(image):
    histogram, bins = np.histogram(image, bins=256, range=(0, 256))
    n = 255
    exposure = 0
    b = np.arange(0, 256)
    # exposure = np.prod(histogram, b)
    for i in range(256):
        # exposure += (bins[i]+bins[i+1])/2 * histogram[i] / n
        exposure += (b[i] * histogram[i]) / n
    return exposure


def flatness_measure(histogram):
   # Calculate the arithmetic mean of the histogram
    arithmetic_mean = np.mean(histogram)
    
    # Calculate the geometric mean of the histogram
    histogram_nonzero = histogram[histogram != 0]
    geometric_mean = np.exp(np.mean(np.log(histogram_nonzero)))
    return geometric_mean/arithmetic_mean

# Function to calculate histogram spread
def histogram_spread(histogram):
    cdf = np.cumsum(histogram)
    ncdf = cdf / np.sum(histogram)
    bin_75 = np.argwhere(ncdf >= 0.75)[0][0]
    bin_25 = np.argwhere(ncdf >= 0.25)[0][0]

    HS = (bin_75 - bin_25)/255
    return HS


def calculate_HFM_HS(im):
    histogram = cv.calcHist([im], [0], None, [256], [0, 256])
    flatness = flatness_measure(histogram)
    spread = histogram_spread(histogram)
    return flatness, spread

orig_exposure = calculate_exposure(image)
r = 1 if orig_exposure < 0.5 else -1


def fitness_function(solution, solution_idx):
    a,c = solution
    global image, r
    enhanced = imageTransform(image, a, c)
    hfm, hs = calculate_HFM_HS(enhanced)
    entropy = skimage.measure.shannon_entropy(enhanced)
    # exp = calculate_exposure(enhanced)
    print("hfm: {:.2f} hs: {:.2f} entropy: {:.2f}".format(hfm, hs, entropy))
    # fitness = hfm * hs * entropy * (exp**r)
    fitness = 100*hfm + 200*hs  + 300*entropy
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

a__ = np.linspace(0, 1.5, 500, endpoint=True)
c__ = np.linspace(0, 1, 500, endpoint=True)

ga_instance = pygad.GA(num_generations=50,
                        num_parents_mating=5, 
                        fitness_func=fitness_function,
                        # init_range_high=1,
                        # init_range_low=0.0,
                        sol_per_pop=10, 
                        parent_selection_type="tournament",
                        keep_parents=3,
                        num_genes=2,
                        mutation_type="random",
                        mutation_by_replacement=True,
                        mutation_percent_genes=10,

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

cv.imwrite("results/res_{now}.png".format(now=datetime.now()), final_image)

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


