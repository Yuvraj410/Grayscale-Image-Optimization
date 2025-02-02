#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np



# Load image
image = cv2.imread('image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define values for g, m, a, and c
g = 128
m = 128
a = 1.5
c = 1.5

# Apply transformation to each pixel in the image
transformed = np.zeros_like(gray)



def transform(x, g, m, a, c):
    y = (x - m) * a + m
    y = np.clip(y, 0, 255)
    if x < g:
        y = y * (x / g) ** c
    else:
        y = y * ((255 - x) / (255 - g)) ** c
    y = np.clip(y, 0, 255)
    return y

for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        transformed[i,j] = transform(gray[i,j], g, m, a, c)

# Display original and transformed images side by side
cv2.imshow('Original', gray)
cv2.imshow('Transformed', transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()


