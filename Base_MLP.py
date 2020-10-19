# -*- coding: utf-8 -*-
import util
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load csv files
train1_x, train1_y = util.load_csv(util.train_1_filepath)
train2_x, train2_y = util.load_csv(util.train_2_filepath)
valid1_x, valid1_y = util.load_csv(util.val_1_filepath)
valid2_x, valid2_y = util.load_csv(util.val_2_filepath)
test1_x, test1_y = util.load_csv(util.test_with_label_1_filepath)
test2_x, test2_y = util.load_csv(util.test_with_label_2_filepath)



def eval_dataset(dataset_num, model):
    
    scores = []
    
    print(f'\nTraining with dataset {dataset_num}..')
    if dataset_num == 1:
        
        scores.append(model.score(train1_x, train1_y))
        scores.append(model.score(valid1_x, valid1_y))
        scores.append(model.score(test1_x, test1_y))
        
    elif dataset_num == 2:
        
        scores.append(model.score(train2_x, train2_y))
        scores.append(model.score(valid2_x, valid2_y))
        scores.append(model.score(test2_x, test2_y))
     
    print(f'Scores with dataset {dataset_num} using Base-MLP')
    print("---------------------------------------")
    print(f'Training Score:     {scores[0]}')
    print(f'Validating Score:   {scores[1]}')
    print(f'Test Score:         {scores[2]}\n')
    

def plot_confusion_matrix(model, y_target, y_predict):
    
    print("Confusion matrix:")
    c_matrix = confusion_matrix(y_target, y_predict)
    print(f'{c_matrix}')
    
    return c_matrix
    
    
def print_model_details(y_target, y_predict):
    report = classification_report(y_target, y_predict)
    print(f'\n{report}\n')
    

def run():
    
    print("Performing Base MLP")
    print("-------------------")
    
    # Train with dataset
    mlp_clfr = MLPClassifier(activation="logistic", solver="sgd") #100 neurons by default
    mlp_model1 = mlp_clfr.fit(train1_x, train1_y)
    mlp_model2 = mlp_clfr.fit(train2_x, train2_y)
    
    # Predict trained model with dataset
    y_predict1 = mlp_model1.predict(test1_x)
    y_predict2 = mlp_model2.predict(test2_x)
    
    # Evaluate score on dataset
    eval_dataset(1, mlp_model1)
    
    # Plot confusion matrix
    c_matrix1 = plot_confusion_matrix(mlp_model1, test1_y, y_predict1)
    
    # Print precision, recall, f1-score, accuracy, macro-avg f1, weighted-avg f1
    print_model_details(test1_y, y_predict1)
    
    # Repeat steps for dataset 2
    eval_dataset(2, mlp_model2)
    test2_y_predict = mlp_model2.predict(test2_x)
    c_matrix2 = plot_confusion_matrix(mlp_model2, test2_y, y_predict2)
    print_model_details(test2_y, y_predict2)
    
    
    # Output results into file
    util.write_csv("./output/Base-MLP-DS1.csv", test1_y, y_predict1, c_matrix1)
    util.write_csv("./output/Base-MLP-DS2.csv", test2_y, y_predict2, c_matrix2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    