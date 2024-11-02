# train/train_classic_models.py
from models.classic_models import get_classic_models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import pandas as pd

def train_classic_models(X_train, X_test, y_train, y_test):
    models = get_classic_models()
    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        print(f"{model_name} - Classification Report:\n", classification_report(y_test, y_pred))
    
    return pd.DataFrame(results).transpose()
