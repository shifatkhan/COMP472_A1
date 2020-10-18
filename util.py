# -*- coding: utf-8 -*-
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from pathlib import Path
import pandas as pd
import sklearn
import numpy
import csv
import util

#Make output cleaner
import warnings
warnings.filterwarnings('ignore') 


#read files
train_1_filepath = './dataset/Assig1-Dataset/train_1.csv'
train_2_filepath = './dataset/Assig1-Dataset/train_2.csv'
val_1_filepath = './dataset/Assig1-Dataset/val_1.csv'
val_2_filepath = './dataset/Assig1-Dataset/val_2.csv'
test_with_label_1_filepath = './dataset/Assig1-Dataset/test_with_label_1.csv'
test_with_label_2_filepath = './dataset/Assig1-Dataset/test_with_label_2.csv'

#output path
output_dir = "./output"

"""
Utility method for reading csv files.
To use this method, here is usage example below:
    x,y = load_csv(train_1_filepath)

Input is file path of the csv file
Returns a tuple of the attributes and the index
"""
def load_csv(filepath):
    temp = [] #1d arr
    training_data = [] #2d array

    try:    
        with open(filepath, 'r') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            
            #create 2d array of attributes and index
            for line in lines:
                temp = []
                
                for item in line:
                    temp.append(int(item))
                    
                training_data.append(temp)
            
            #seperate index from binary
            x = []
            y = []
            
            for line in training_data:
                attr = line[:-1]
                output = line[-1]
                x.append(attr)
                y.append(output)
                
            return x,y
        
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    
"""
Utility method for writing csv files.

Parameters
----------
filepath : str
    Path to file we're going to save to (eg. "./output/Base-DT-DS1.csv")
Y_test : int[]
    The Y test inputs gotten from test_with_label
Y_pred : int[]
    The predicated Ys from your model
confusion_matrix : JSON
    Gotten from metrics.confusion_matrix() function
        
"""
def write_csv(filepath, Y_test, Y_pred, confusion_matrix):
    try:
        # Create output dir if it doesn't exist.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # newline="" because there are empty lines between rows.
        with open(filepath, mode='w', newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # WRITE PART A)
            csv_writer.writerow([""]) # newline
            csv_writer.writerow(["a) Predicted classes"]) # Must be an Iterator, so put []
            
            for i in range(len(Y_pred)):
                csv_writer.writerow([i+1, Y_pred[i]])
                
            # WRITE PART B)
            csv_writer.writerow([""]) # newline
            csv_writer.writerow(["b) Confusion Matrix"])
            for i in range(len(confusion_matrix)):
                csv_writer.writerow(confusion_matrix[i])
                
            # WRITE PART C) & D)
            csv_writer.writerow([""]) # newline
            csv_writer.writerow(["c) & d) Classification report for classifier"])
                
    except FileNotFoundError:
            print(f"File not found: {filepath}")
            
    # WRITE PART C) & D) cont'd
    data_frame = pd.DataFrame(metrics.classification_report(Y_test, Y_pred, output_dict=True)).transpose()
    data_frame.to_csv(filepath, mode="a")
