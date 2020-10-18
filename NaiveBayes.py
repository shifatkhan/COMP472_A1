# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:21:35 2020

@author: William Ngo
"""
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import util

"""
This functions will be used by the main.py to execute the gaussian naives bayes
algorithm.
"""
def run_naivebayes():
    output_filepath1 = "./output/GNB-DS1.csv"
    output_filepath2 = "./output/GNB-DS2.csv"
    
    run_dataset(util.train_1_filepath, util.test_with_label_1_filepath, output_filepath1)
    run_dataset(util.train_2_filepath, util.test_with_label_2_filepath, output_filepath2)
    
"""
This function will execute the gaussion naives bayes algorithm on the dataset
located in the given filepath string. It will generate an output file in the
output folder located inside this project
"""    
def run_dataset(filepath_train, filepath_test, filepath_output):
        
    x_train, y_train = util.load_csv(filepath_train)
    x_test, y_test = util.load_csv(filepath_test)
    
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    
    train_accuracy = gnb.score(x_train, y_train)
    test_accuracy = metrics.accuracy_score(y_test, y_pred)
    
    #confusion matrix
    cmatrix = metrics.confusion_matrix(y_test, y_pred)
    metrics.plot_confusion_matrix(gnb, x_test, y_test)
    
    #evalution
    classification_report = metrics.classification_report(y_test, y_pred)
    
    #output file
    util.write_csv(filepath_output, y_test, y_pred, cmatrix)
    
    #Print result to console
    print_result(gnb, train_accuracy, test_accuracy, y_pred, cmatrix, classification_report)

"""
Prints result to console. For debugging purposes
"""
def print_result(gnb, train_acc, test_acc, y_pred, cmatrix, classification_report):
    print("\nResult") 
    print(f"Train Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")
    
    print("")
    print("row number of the instance, index of the predicted class of that instance")
    for i in range(len(y_pred)):
        print(f"{i}, {y_pred[i]}")
        
    print("")    
    print("Confusion Matrix")
    print(cmatrix)
    
    print("")
    print("Classification report for classifier %s:\n%s\n" % (gnb, classification_report))
    
run_naivebayes()