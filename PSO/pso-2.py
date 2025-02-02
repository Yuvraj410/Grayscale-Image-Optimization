#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

image = cv2.imread('img/lena_condec.jpg', cv2.IMREAD_GRAYSCALE)

def pso_contrast_enhancement(image):
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB channels
    l, a, b = cv2.split(lab)

    # Define histogram equalization function
    def histeq(channel):
        hist, bins = np.histogram(channel.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_norm = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf_norm = cdf_norm.astype('uint8')
        return cdf_norm[channel]
    
    # Apply histogram equalization to L channel
    l_eq = histeq(l)

    # Perform PSO optimization
    num_particles = 50
    max_iterations = 100
    c1 = 0.5
    c2 = 0.5
    w = 0.9
    vmin = 0
    vmax = 255

    def objective(x):
        y = (l_eq - x[0]) * (x[1] - x[0]) / (x[3] - x[2]) + x[0]
        y = np.clip(y, vmin, vmax)
        return np.mean((y - l_eq) ** 2)

    x_min = np.array([l_eq.min(), l_eq.min(), l_eq.mean(), l_eq.max()])
    x_max = np.array([l_eq.mean(), l_eq.max(), l_eq.mean(), l_eq.max()])

    particles = np.random.uniform(x_min, x_max, size=(num_particles, 4))
    pbest = particles.copy()
    pbest_fitness = np.array([objective(p) for p in pbest])
    gbest_index = np.argmin(pbest_fitness)
    gbest = pbest[gbest_index]
    gbest_fitness = pbest_fitness[gbest_index]

    for i in range(max_iterations):
        r1 = np.random.rand(num_particles, 4)
        r2 = np.random.rand(num_particles, 4)
        v = w * (particles - pbest) + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)
        particles = particles + v
        particles = np.clip(particles, x_min, x_max)
        fitness = np.array([objective(p) for p in particles])
        update = fitness < pbest_fitness
        pbest[update] = particles[update]
        pbest_fitness[update] = fitness[update]
        if np.min(pbest_fitness) < gbest_fitness:
            gbest_index = np.argmin(pbest_fitness)
            gbest = pbest[gbest_index]
            gbest_fitness = pbest_fitness[gbest_index]

    # Apply contrast stretching to L channel
    l_stretch = np.interp(l, [gbest[0], gbest[1]], [gbest[2], gbest[3]])

    # Merge LAB channels
    lab_eq = cv2.merge((l_eq, a, b))
    lab_stretch = cv2.merge((l_stretch, a, b))

    # Convert back to RGB color space
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    rgb_stretch = cv2.cvtColor(lab_stretch, cv2.COLOR_LAB2BGR)

