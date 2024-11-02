# main.py
import torch
import pandas as pd
from data.preprocess import load_data, preprocess_data, split_and_scale_data
from train.train_classic_ml import train_classic_models
from train.train_deep_model import train_deep_model
from train.utils import plot_classification_report, plot_confusion_matrix

# Load and preprocess data
data = load_data("data/dataset_sdn.csv")
data = preprocess_data(data)
X_train, X_test, y_train, y_test = split_and_scale_data(data)

# Train and evaluate classic models
results = train_classic_models(X_train, X_test, y_train, y_test)
print("Classic Models Performance:\n", results)

# Train deep learning model
model = train_deep_model(X_train, y_train, input_dim=X_train.shape[1])

# Evaluate deep learning model
y_pred = (model(torch.tensor(X_test, dtype=torch.float32)) > 0.5).float().numpy()
plot_classification_report(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, labels=['Benign', 'Malicious'])
