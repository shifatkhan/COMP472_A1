# -*- coding: utf-8 -*-
import sklearn
import csv
import numpy


#read files
train_1_filepath = './dataset/Assig1-Dataset/train_1.csv'
train_2_filepath = './dataset/Assig1-Dataset/train_2.csv'
val_1_filepath = './dataset/Assig1-Dataset/val_1.csv'
val_2_filepath = './dataset/Assig1-Dataset/val_2.csv'
test_with_label_1_filepath = './dataset/Assig1-Dataset/test_with_label_1.csv'
test_with_label_2_filepath = './dataset/Assig1-Dataset/test_with_label_2.csv'

"""
Utility method for reading csv files.
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
    
    
# To use this method, here is usage example below
# x,y = load_csv(train_1_filepath)
