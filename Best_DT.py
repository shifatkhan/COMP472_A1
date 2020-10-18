# -*- coding: utf-8 -*-

from sklearn import tree
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import GridSearchCV # Cross validation for getting the best DT
import util

# TODO: Move to util?
def print_debug(data_index, clf, Y_pred, confusion_matrix, classification_report):
    print("\na)")
    for i in range(len(Y_pred)):
        print(f"{i}, {Y_pred[i]}")
        
    print("\nb)")
    print("Confusion Matrix")
    print(confusion_matrix)
    
    print("\nc) & d)")
    print("Classification report for classifier %s:\n%s\n"
          % (clf, classification_report))

def run():
    # ========= DATASET 1 ========= #
    filepath = "./output/Best-DT-DS1.csv"

    X_train, Y_train = util.load_csv(util.train_1_filepath)
    X_test, Y_test = util.load_csv(util.test_with_label_1_filepath)
    
    # Initial DT
    SEED = 1
    clf = tree.DecisionTreeClassifier(random_state=SEED)
    
    print(f"BEST DT dataset1 ===========================================")
    print("Initial DT parameters:")
    print(clf.get_params())
    
    # Define the hyperparameters
    parameters = {
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [10, None],
        'min_samples_split' : [2, 3, 4],
        'min_impurity_decrease' : [0, 0.05],
        'class_weight' : [None, 'balanced']
        }
    
    # Find best parameters.
    grid_dt = GridSearchCV(estimator = clf, param_grid = parameters, n_jobs = -1, cv = 3)
    grid_dt.fit(X_train, Y_train)
    
    best_params = grid_dt.best_params_
    
    # Get new best estimator (classifier)
    clf = grid_dt.best_estimator_
    print("\nBest DT parameters:")
    print(clf.get_params())
    
    # Test/Predict
    Y_pred = clf.predict(X_test)
    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    metrics.plot_confusion_matrix(clf, X_test, Y_test)
    # Evaluation
    classification_report = metrics.classification_report(Y_test, Y_pred)
    # Debug print
    print_debug(1, clf, Y_pred, confusion_matrix, classification_report)
    # Save
    util.write_csv(filepath, Y_test, Y_pred, confusion_matrix)
    
    # ========= DATASET 2 ========= #
    filepath = "./output/Best-DT-DS2.csv"

    X_train, Y_train = util.load_csv(util.train_2_filepath)
    X_test, Y_test = util.load_csv(util.test_with_label_2_filepath)
    
    # Initial DT
    clf = tree.DecisionTreeClassifier(random_state=SEED)
    
    print(f"BEST DT dataset2 ===========================================")
    print("Initial DT parameters:")
    print(clf.get_params())
    
    # Find best parameters. CV = 15 because the dataset is much bigger.
    grid_dt = GridSearchCV(estimator = clf, param_grid = parameters, n_jobs = -1, cv = 7)
    grid_dt.fit(X_train, Y_train)
    
    best_params = grid_dt.best_params_
    
    # Get new best estimator (classifier)
    clf = grid_dt.best_estimator_
    
    # Test/Predict
    Y_pred = clf.predict(X_test)
    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
    metrics.plot_confusion_matrix(clf, X_test, Y_test)
    # Evaluation
    classification_report = metrics.classification_report(Y_test, Y_pred)
    # Debug print
    print_debug(2, clf, Y_pred, confusion_matrix, classification_report)
    print("\nBest DT parameters:")
    print(clf.get_params())
    # Save
    util.write_csv(filepath, Y_test, Y_pred, confusion_matrix)
    
    
    
# DEBUG--------------------------------------------------------------------
#run()