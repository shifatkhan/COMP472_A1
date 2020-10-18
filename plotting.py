# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 15:21:12 2020

@author: William Ngo
"""

import util
import matplotlib.pyplot as plt
from collections import Counter

train1 = util.train_1_filepath
train2 = util.train_2_filepath

attr1,classes1 = util.load_csv(train1)
attr2,classes2 = util.load_csv(train2)

def drawPlotInstance(listOfY):
    #listOfY contains the last column of the training data set
    counts = Counter(listOfY) #counter counts the number of instances of each class
    x_array = []
    y_count_array = []
    length = len(counts)
    
    #building x axis
    for i in range(length):
        x_array.append(i)
    
    #building y axis
    for i in range(length):
        y_count_array.append(counts[i])
    
    plt.plot(x_array, y_count_array)
    plt.ylabel("Number of instances")
    plt.xlabel("Training set 2")
    
#drawPlotInstance(classes2)