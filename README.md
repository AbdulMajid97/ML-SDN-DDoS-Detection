# DDoS Detection in SDN Environments

A machine learning and deep learning-based solution to detect Distributed Denial of Service (DDoS) attacks in Software-Defined Networking (SDN) environments. This project leverages both classical ML models (Random Forest, SVM, Naive Bayes, etc.) and a custom deep learning model in PyTorch to identify malicious traffic patterns, achieving high accuracy in distinguishing benign from malicious flows.

## Project Structure

DDoS_Detection_Project/ │ ├── config/ │ └── config.yaml # Configuration file for parameters, paths, etc. │ ├── data/ │ ├── dataset/ │ │ └── dataset_sdn.csv # Raw dataset file │ ├── preprocess.py # Script for data loading and preprocessing │ └── init.py # Allows data to be imported as a module │ ├── models/ │ ├── classic_models.py # Contains definitions for classic ML models (Random Forest, etc.) │ ├── dl_ddos_classifier.py # PyTorch deep learning model definition (DDoSClassifier class) │ └── init.py # Allows models to be imported as a module │ ├── train/ │ ├── train_classic_models.py # Script for training and evaluating classic ML models │ ├── train_deep_model.py # Script for training the deep learning model with TensorBoard │ └── utils.py # Utility functions for metrics, logging, and plotting │ ├── notebooks/ │ └── initial_exploration.ipynb # Original notebook with initial code and analysis │ ├── runs/ # TensorBoard logs (populated during training) │ ├── results/ │ ├── evaluation_report.txt # File to save final model evaluation metrics │ ├── confusion_matrix.png # Confusion matrix plot │ └── classification_report.png # Classification report plot │ ├── main.py # Entry point to run the project ├── requirements.txt # Project dependencies └── README.md # Project documentation


## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- [PyTorch](https://pytorch.org/get-started/locally/) (with CUDA support if GPU acceleration is available)

### Install Dependencies

1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/DDoS_Detection_Project.git
   cd DDoS_Detection_Project
Install the required Python packages:
pip install -r requirements.txt
