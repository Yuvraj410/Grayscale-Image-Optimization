import cv2
import numpy as np
import random

# Load the image
image = cv2.imread("../img/balloons.png", 0)

# Define the fitness function
def fitness_function(solution):
    contrast_stretched_image = np.interp(image, np.arange(256), solution)
    mse = np.mean((contrast_stretched_image - np.mean(contrast_stretched_image)) ** 2)
    fitness_score = 1 / (1 + mse)
    return fitness_score

# Define the genetic algorithm
def genetic_algorithm(population_size, num_generations, mutation_rate):
    # Define the initial population
    population = []
    for i in range(population_size):
        solution = np.sort(np.random.rand(256) * 255)
        population.append(solution)

    # Iterate over the generations
    for generation in range(num_generations):
        # Evaluate the fitness of each solution
        fitness_scores = []
        for solution in population:
            fitness_scores.append(fitness_function(solution))

        # Select the best solutions for breeding
        breeding_pool = []
        for i in range(population_size // 2):
            parent1 = population[fitness_scores.index(max(fitness_scores))]
            fitness_scores[fitness_scores.index(max(fitness_scores))] = -1
            parent2 = population[fitness_scores.index(max(fitness_scores))]
            fitness_scores[fitness_scores.index(max(fitness_scores))] = -1
            breeding_pool.append((parent1, parent2))

        # Breed new solutions
        population = []
        for parents in breeding_pool:
            parent1, parent2 = parents
            child = np.zeros(max(len(parent1), len(parent2)))
            crossover_point = random.randint(1, len(child) - 2)
            child[:crossover_point] = parent1[:crossover_point]
            child[crossover_point:] = parent2[crossover_point:]
            population.append(child)

        # Mutate some of the solutions
        for i in range(population_size):
            for j in range(len(population[i])):
                if random.random() < mutation_rate:
                    population[i][j] += random.randint(-10, 10)
                    population[i][j] = np.clip(population[i][j], 0, 255)

        # Print the best solution of this generation
        best_solution = population[fitness_scores.index(max(fitness_scores))]
        print("Generation {}: Fitness = {}".format(generation, max(fitness_scores)))

    # Apply the best solution to the image
    contrast_stretched_image = np.interp(image, np.arange(256), best_solution)
    cv2.imwrite("output_image.png", contrast_stretched_image)

# Run the genetic algorithm
genetic_algorithm(population_size=50, num_generations=100, mutation_rate=0.05)
