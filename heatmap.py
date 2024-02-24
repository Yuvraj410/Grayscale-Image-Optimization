#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:24:50 2022

@author: daniyal
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

fitnesses = np.load("arrays/fitnesses.npy").round(decimals=4)
hss = np.load("arrays/hss.npy")
hfms = np.load("arrays/hfms.npy")
entropies  = np.load("arrays/entropies.npy")



# plt.subplot(2,2,1)
# plt.imshow( fitnesses, interpolation = 'nearest',cmap="rainbow")
# plt.title('Fitness HeatMap Using Matplotlib')
 
# plt.subplot(2,2,2)
# plt.imshow( entropies, interpolation = 'nearest',cmap="rainbow")
# plt.title('Entropy HeatMap Using Matplotlib Library')
 
# plt.subplot(2,2,3)
# plt.imshow( hfms, interpolation = 'nearest',cmap="rainbow")
# plt.title('HFM HeatMap Using Matplotlib Library')
 
# plt.subplot(2,2,4)
# plt.imshow( hss, interpolation = 'nearest',cmap="rainbow")
# plt.title('HS HeatMap Using Matplotlib Library')
 
# plt.tight_layout()
 
# plt.show()



a__ = np.load("arrays/a__.npy")

c__ = np.load("arrays/c__.npy")
a__ = a__.round(decimals = 2)



fig, ax = plt.subplots()
im = ax.imshow(fitnesses)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(c__)), labels=c__)
ax.set_yticks(np.arange(len(a__)), labels=a__)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(a__)):
    for j in range(len(c__)):
        text = ax.text(j, i, fitnesses[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Fitness values with respect to a and c")
fig.tight_layout()
plt.show()