# -*- coding: utf-8 -*-

from sklearn import tree
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import util

def run():
    filepath = "./output/Base-DT-DS1.csv"

    X_train, Y_train = util.load_csv(util.train_1_filepath)
    X_test, Y_test = util.load_csv(util.test_with_label_1_filepath)
    
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    
    # Train
    clf = clf.fit(X_train, Y_train)
    
    # Test/Predict
    Y_pred = clf.predict(X_test)
            
    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    metrics.plot_confusion_matrix(clf, X_test, Y_test)
    
    # Evaluation
    classification_report = metrics.classification_report(Y_test, Y_pred)
    
    # DEBUG
    print("BASE DT ========================================================");
    print(f"Train Accuracy: {clf.score(X_train, Y_train)}");
    print(f"Test Accuracy: {metrics.accuracy_score(Y_test, Y_pred)}");
    
    print("\na)")
    for i in range(len(Y_pred)):
        print(f"{i}, {Y_pred[i]}")
        
    print("\nb)")
    print("Confusion Matrix")
    print(confusion_matrix)
    
    print("\nc) & d)")
    print("Classification report for classifier %s:\n%s\n"
          % (clf, classification_report))
    
    util.write_csv(filepath, Y_test, Y_pred, confusion_matrix)
    
# DEBUG--------------------------------------------------------------------
run()