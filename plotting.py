# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:21:12 2020

@author: William Ngo
"""

import util
import matplotlib.pyplot as plt
from collections import Counter

train1 = util.train_1_filepath
train2 = util.train_2_filepath

attr1,classes1 = util.load_csv(train1)
attr2,classes2 = util.load_csv(train2)

def drawPlotInstance(classes1, classes2):
    #listOfY contains the last column of the training data set
    counts_1 = Counter(classes1) #counter counts the number of instances of each class
    counts_2 = Counter(classes2)
    x_array_1 = []
    x_array_2 = []
    y_count_array_1 = []
    y_count_array_2 = []
    length_1 = len(counts_1)
    length_2 = len(counts_2)
    
    #==================== dataset 1
    #building x axis
    for i in range(length_1):
        x_array_1.append(i)
    
    #building y axis
    for i in range(length_1):
        y_count_array_1.append(counts_1[i])
    
    #==================== dataset 2
    #building x axis
    for i in range(length_2):
        x_array_2.append(i)
    
    #building y axis
    for i in range(length_2):
        y_count_array_2.append(counts_2[i])
        
    #plotting
    figures, axes = plt.subplots(2)
    
    axes[0].plot(x_array_1, y_count_array_1)
    #axes[0].set_title("Dataset 1")
    axes[0].set(ylabel="Dataset 1")
    axes[0].set(xlabel="Classes")
    
    axes[1].plot(x_array_2, y_count_array_2)
    #axes[1].set_title("Dataset 2")
    axes[1].set(ylabel="Dataset 2")
    axes[1].set(xlabel="Classes")
    
drawPlotInstance(classes1, classes2)