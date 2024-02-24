import cv2 as cv
import numpy as np
from pyswarm import pso
import skimage

# Read the input color image
orig_img = cv.imread('/home/daniyal/image-processing/img/balloons_lowcontrast_lowbrightness.png')

# Convert the input image to grayscale
gray_img = cv.cvtColor(orig_img, cv.COLOR_BGR2GRAY)


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


arith_mean_arr = local_am(gray_img)

geom_mean_arr = local_gm(gray_img, 3)

   
def calculateGain(img,i, j, n):
    imgH, imgW =  np.shape(img)
    return geom_mean_arr[i, j]/arith_mean_arr[i, j]

 
def imageTransform(img, params):
    print(params)
    a, c = params
    imgH, imgW =  np.shape(img)
    temp = np.zeros(np.shape(img), dtype="float32")
    global arith_mean_arr, geom_mean_arr
    for i in range(0, imgH):
        for j in range(0, imgW):
            m = arith_mean_arr[i, j]
            gain = calculateGain(img, i, j, 3)
            temp[i][j] = gain*(img[i][j]-c*m) + m**a
    # max_val = np.amax(temp)
    # if max_val>255:
    #     med = np.median(temp)
    #     diff = med - 128
    #     temp = temp - diff
        
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
    r = np.power(histogram, 1/255)
    return np.prod(r, where=r>0) / np.mean(histogram)

def histogram_spread(histogram):
    first_quartile = np.percentile(histogram, 25)
    third_quartile = np.percentile(histogram, 75)
    return (third_quartile - first_quartile) / 255

def calculate_HFM_HS(im):
    histogram, _ = np.histogram(im, bins=256, range=(0, 256))
    flatness = flatness_measure(histogram)
    spread = histogram_spread(histogram)
    return flatness, spread

# Define the fitness function for PSO
def fitness_func(params):
    print(params)
    a, c = params
    global gray_img
    # Convert the grayscale image to color
    color_img = cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR)
    
    # Apply the enhancement algorithm using the PSO parameters
    enhanced_img = imageTransform(gray_img, params) 
    
    # Calculate the quality metric of the enhanced image
    # quality_metric = calculate_quality_metric(img, enhanced_img)
    
    # Return the quality metric as the fitness value
    # return quality_metric
    hfm, hs = calculate_HFM_HS(gray_img)
    entropy = skimage.measure.shannon_entropy(gray_img)
    fitness = hfm*hs*entropy
    return 1/fitness

# Define the bounds of the PSO parameters
# lb = [0, 0, 0, 0]   # lower bounds
# ub = [255, 255, 255, 255]   # upper bounds
lb = [0, 0]   # lower bounds
ub = [1, 1.5]   # upper bounds

# Define the PSO parameters
n_particles = 20
n_iterations = 25
inertia_weight = 0.8
cognitive_weight = 1.5
social_weight = 1.5

# Run the PSO algorithm to find the optimal set of parameters
best_params, best_fitness = pso(fitness_func, lb, ub, swarmsize=n_particles, maxiter=n_iterations, 
                                phip=cognitive_weight, phig=social_weight, omega=inertia_weight)

# Apply the enhancement algorithm to the input image using the optimal set of parameters
enhanced_img = imageTransform(gray_img, best_params)

# Save the enhanced image
cv.imwrite('enhanced_image1.jpg', enhanced_img)
# -*- coding: utf-8 -*-

