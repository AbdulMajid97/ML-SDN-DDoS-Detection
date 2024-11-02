# data/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Handle missing values
    mean_rx_kbps = data['rx_kbps'].mean()
    mean_tot_kbps = data['tot_kbps'].mean()
    data.fillna({'rx_kbps': mean_rx_kbps, 'tot_kbps': mean_tot_kbps}, inplace=True)

    # Label encode categorical features
    label_encoder = {col: LabelEncoder() for col in ['src', 'dst', 'Protocol']}
    for col, le in label_encoder.items():
        data[col] = le.fit_transform(data[col])

    return data

def split_and_scale_data(data):
    X = data.drop(columns='label')
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
