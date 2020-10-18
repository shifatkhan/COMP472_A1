# -*- coding: utf-8 -*-

from sklearn import tree
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from pathlib import Path
import pandas as pd
import csv
import util

import warnings
warnings.filterwarnings('ignore') #Make output cleaner

filename = "./output/Base-DT-DS1.csv"

X_train, Y_train = util.load_csv(util.train_1_filepath)
X_test, Y_test = util.load_csv(util.test_with_label_1_filepath)

clf = tree.DecisionTreeClassifier(criterion="entropy")

# Train
clf = clf.fit(X_train, Y_train)
print(f"Train Accuracy: {clf.score(X_train, Y_train)}");

# Test/Predict
Y_pred = clf.predict(X_test)
print(f"Test Accuracy: {metrics.accuracy_score(Y_test, Y_pred)}");

print("\na)")

for i in range(len(Y_pred)):
    print(f"{i}, {Y_pred[i]}")
        
print("\nb)")
print("Confusion Matrix")
confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
metrics.plot_confusion_matrix(clf, X_test, Y_test)
print(confusion_matrix)

print("\nc) & d)")
classification_report = metrics.classification_report(Y_test, Y_pred)
print("Classification report for classifier %s:\n%s\n"
      % (clf, classification_report))


# OUTPUT
try: 
    Path("./output").mkdir(parents=True, exist_ok=True)
    # newline="" because there are empty lines between rows.
    with open(filename, mode='w', newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write part A
        csv_writer.writerow([""]) # newline
        csv_writer.writerow(["a) Predicted classes"]) # Must be an Iterator, so put []
        
        for i in range(len(Y_pred)):
            csv_writer.writerow([i+1, Y_pred[i]])
            
        # Write part B
        csv_writer.writerow([""]) # newline
        csv_writer.writerow(["b) Confusion Matrix"])
        for i in range(len(confusion_matrix)):
            csv_writer.writerow(confusion_matrix[i])
            
        # Write part C & D
        csv_writer.writerow([""]) # newline
        csv_writer.writerow(["c) & d) Classification report for classifier"])
            
except FileNotFoundError:
        print(f"File not found: {filename}")
        
data_frame = pd.DataFrame(metrics.classification_report(Y_test, Y_pred, output_dict=True)).transpose()
data_frame.to_csv(filename, mode="a")