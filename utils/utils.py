import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import pickle
import os

def plot_wandb(wandb_login, project_name, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, labels, model_name):
    wandb.init(id=wandb_login, project=project_name)
    wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
    wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_test, y_test)
    if y_pred_proba:
        wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, labels,
                                                            model_name=model_name)
    else:
        wandb.sklearn.plot_summary_metrics(eclf, X_train, y_train, X_test, y_test)
        wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels)
        wandb.sklearn.plot_learning_curve(eclf, X_train, y_train)
        
def calculate_metrics(model, X ,y, y_pred):
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)
    score = model.score(X, y)
    return f1, auc, score

def plot_auc_roc(y, y_pred):
    fpr, tpr, _ = roc_curve(y, y_pred)
    plt.figure(figsize=(16,8) )
    sns.lineplot(fpr,tpr,label="AUC="+str(auc))
    
def confusion_matrix(model, X, y):
    plot_confusion_matrix(model, X, y)
    plt.show()
    
def save_model(model, out_filepath, model_name):
    pickle.dump(model, open(os.path.join(out_filepath, model_name), 'wb'))