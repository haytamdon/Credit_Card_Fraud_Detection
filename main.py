from model.Models import Model
from data_processing.data_processing import *
import sklearn 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from utils.utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os

if __name__=='__main__':
    data = read_data('../creditcard.csv')
    X_train, y_train, X_test, y_test = data_cleaning(data)

    my_model = Model('Logistic regression').get_model()
    my_model.fit(X_train, y_train)
    
    y_pred = my_model.predict(X_test)
    y_pred_proba = my_model.predict_proba(X_test)
    
    f1, auc, score = calculate_metrics(my_model, X_test ,y_test, y_pred)
    
    print("f1 =", f1, "AUC score", auc, "Accuracy", score)
    
    plot_auc_roc(y_test, y_pred)
    confusion_matrix(my_model, X_test, y_test)
    
    save_model(my_model, '../Model_Registry', 'Logistic Regression.sav')
