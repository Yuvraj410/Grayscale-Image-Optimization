#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:00:09 2023

@author: daniyal
"""

import cv2
import numpy as np
import random

# Load the image
img = cv2.imread('/home/daniyal/image-processing/img/balloons_lowcontrast_lowbrightness.png', cv2.IMREAD_GRAYSCALE)

# Define the GA parameters
pop_size = 10         # Population size
num_gen = 1000         # Number of generations
mutation_rate = 0.1   # Mutation rate
lb = 0.0              # Lower bound of enhancement factor
ub = 2.0              # Upper bound of enhancement factor

# Define the fitness function
def fitness_func(f, img):
    enhanced_img = np.clip(f * img, 0, 255).astype(np.uint8)
    fitness = -cv2.Laplacian(enhanced_img, cv2.CV_64F).var()
    return fitness

# Initialize the population
population = np.random.uniform(lb, ub, pop_size)
fitness_values = np.array([fitness_func(f, img) for f in population])
best_idx = np.argmax(fitness_values)

# GA algorithm
for i in range(num_gen):
    # Selection
    sorted_idx = np.argsort(fitness_values)[::-1]
    parents_idx = sorted_idx[:2]
    offspring = population[parents_idx].copy()

    # Crossover
    offspring[0] = (offspring[0] + offspring[1]) / 2.0

    # Mutation
    if random.random() < mutation_rate:
        offspring[0] += np.random.normal(0, 0.1)
        offspring[0] = np.clip(offspring[0], lb, ub)

    # Replace worst individual
    worst_idx = np.argmin(fitness_values)
    population[worst_idx] = offspring[0]
    fitness_values[worst_idx] = fitness_func(offspring[0], img)

    # Update best individual
    if fitness_values[worst_idx] > fitness_values[best_idx]:
        best_idx = worst_idx

    print('Generation', i+1, '- Best Fitness:', -fitness_values[best_idx])

# Apply the enhancement factor to the image
enhanced_img = np.clip(population[best_idx] * img, 0, 255).astype(np.uint8)

# Save the enhanced image
cv2.imwrite('enhanced_image.jpg', enhanced_img)
