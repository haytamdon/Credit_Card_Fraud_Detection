import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import wandb
import pickle

def data_cleaning(data):
    with wandb.init(project="Fraud Detection 1", job_type="preprocess-data") as run:
        X = data.iloc[:,:-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        Split_data = wandb.Artifact(
            "Split_Data", type= "dataset"
        )
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('Credit_Card_Fraud_Detection_Dataset:latest')

        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download()
        run.log_artifact(Split_data)
        
    with wandb.init(project="Fraud Detection 1", job_type="preprocess-data") as run:
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        Oversampled_Data = wandb.Artifact(
            "SMOTE_Oversampled_Data", type= "dataset", description='SMOTE OverSampling'
        )
        raw_data_artifact = run.use_artifact('Credit_Card_Fraud_Detection_Dataset:latest')
        raw_dataset = raw_data_artifact.download()
        run.log_artifact(Oversampled_Data)
        
    with wandb.init(project="Fraud Detection 1", job_type="preprocess-data") as run:
        sc = StandardScaler()
        X_train_res = sc.fit_transform(X_train_res)
        X_test = sc.transform(X_test)
        
        Scaled_data = wandb.Artifact(
            "Scaled_Data", type= "dataset", description='Scaling'
        )
        raw_data_artifact = run.use_artifact('Credit_Card_Fraud_Detection_Dataset:latest')
        raw_dataset = raw_data_artifact.download()
        run.log_artifact(Scaled_data)
    with wandb.init(project="Fraud Detection 1", job_type="preprocess-data") as run:
        pca = PCA(n_components = 5)
        X_train_res = pca.fit_transform(X_train_res)
        X_test = pca.transform(X_test)
        PCA_Data = wandb.Artifact(
            "PCA_Data", type= "dataset", description='PCA dimension reduction with Standard Scaling'
        )
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('Credit_Card_Fraud_Detection_Dataset:latest')

        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download()
        run.log_artifact(PCA_Data)
    pickle.dump(pca, open("../../Data_Processing_files/pca.pkl","wb"))
    pickle.dump(sc, open("../../Data_Processing_files/StandardScaler.sav","wb"))
    return X_train_res, y_train_res, X_test, y_test

def read_data(filepath):
    wandb.login()
    with wandb.init(project="Fraud Detection 1", job_type="load-data") as run:
        data = pd.read_csv(filepath)
        raw_data = wandb.Artifact('Credit_Card_Fraud_Detection_Dataset', 
                                type= ".dataset",
                                description= "Raw Credit Card Dataset")
        run.log_artifact(raw_data)
    return data
