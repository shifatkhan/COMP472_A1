# -*- coding: utf-8 -*-
import util
from sklearn.neural_network import MLPClassifier

# Global Vars
mlp_clfr = MLPClassifier(activation="logistic", solver="sgd") #100 neurons by default

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


def perform_bestMLP():
    
    print("Performing Base MLP")
    print("-------------------")
    
    print("\nTraining with dataset..\n")
    mlp_model1 = mlp_clfr.fit(train1_x, train1_y)
    mlp_model2 = mlp_clfr.fit(train2_x, train2_y)
    
    eval_dataset(1, mlp_model1)
    eval_dataset(2, mlp_model2)
    