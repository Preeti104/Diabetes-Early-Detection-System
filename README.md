# Diabetes Early Detection System

A **Machine-Learning-Based Diabetes Early Detection System** that predicts whether a person is likely to have diabetes using clinical and demographic features. The project compares multiple machine-learning algorithms and selects the best-performing model based on evaluation metrics, particularly F1-score.

## Project Overview

This project uses Python and Scikit-learn to develop and evaluate machine-learning models for early diabetes detection.

The implemented workflow includes:

* Loading and exploring the diabetes dataset
* Performing data preprocessing and preparation
* Splitting data into training and testing sets
* Standardising numerical features
* Training multiple machine-learning classification models
* Comparing model performance using evaluation metrics
* Identifying the best-performing model
* Saving the selected trained model as a `.pkl` file

## Machine Learning Models

The system evaluates the following classification algorithms:

1. **Logistic Regression**
2. **Random Forest**
3. **Support Vector Machine (SVM)**
4. **XGBoost**

The models are compared using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Classification Report

The model with the highest **F1-score** is selected as the best-performing model.

## Project Files

```text
Diabetes-Early-Detection-System/
│
├── Diabetes Early Detection System.ipynb
├── Diabetes Dataset.xlsx
├── XGBoost_model.pkl
└── README.md
```

### File Description

| File                                    | Description                                                                   |
| --------------------------------------- | ----------------------------------------------------------------------------- |
| `Diabetes Early Detection System.ipynb` | Jupyter Notebook containing data preprocessing, model training and evaluation |
| `Diabetes Dataset.xlsx`                 | Dataset used for training and testing the machine-learning models             |
| `XGBoost_model.pkl`                     | Saved trained XGBoost model                                                   |
| `README.md`                             | Project documentation and setup instructions                                  |

## Requirements

Install the following software before running the project:

* Python 3.10 or later
* Jupyter Notebook or JupyterLab
* pip

### Python Libraries

The project requires:

```text
pandas
numpy
scikit-learn
xgboost
openpyxl
joblib
matplotlib
seaborn
```

## Installation

### 1. Clone or Download the Project

Download the complete project folder to your computer.

Open **Command Prompt**, PowerShell, or a terminal inside the project directory.

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install Required Libraries

```bash
pip install pandas numpy scikit-learn xgboost openpyxl joblib matplotlib seaborn jupyter
```

Alternatively, install everything together using:

```bash
pip install -r requirements.txt
```

if a `requirements.txt` file is provided.

## Running the Project

### Step 1: Start Jupyter Notebook

Run:

```bash
jupyter notebook
```

### Step 2: Open the Notebook

Open:

```text
Diabetes Early Detection System.ipynb
```

### Step 3: Check the Dataset Path

Make sure:

```text
Diabetes Dataset.xlsx
```

is located in the same project directory as the notebook.

If the notebook uses a relative path, no path modification should be required.

### Step 4: Run the Notebook

Run the notebook cells from top to bottom.

The notebook performs the following process:

```text
Dataset
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Saved .pkl Model
```

## Dataset

The project uses an Excel-based diabetes dataset.

The dataset is loaded using Pandas and prepared for machine-learning classification.

The target variable represents the diabetes outcome, where the model learns to distinguish between:

* **Non-diabetic**
* **Diabetic**

The notebook uses an **80% training and 20% testing split** with stratification to maintain the class distribution.

A fixed `random_state=42` is used to improve reproducibility.

## Model Training

Four machine-learning classification algorithms are trained and evaluated.

### Logistic Regression

Provides a baseline classification model and estimates the probability of diabetes based on the input features.

### Random Forest

Uses multiple decision trees to improve classification performance and reduce dependence on a single decision tree.

### Support Vector Machine

Uses an RBF kernel to identify complex relationships between input features and diabetes outcomes.

### XGBoost

Uses gradient-boosted decision trees and is included as a high-performance classification approach.

## Model Evaluation

The models are evaluated using several metrics.

**Accuracy** measures the overall proportion of correct predictions.

**Precision** measures how many predicted diabetic cases are actually diabetic.

**Recall** measures how many actual diabetic cases are correctly identified.

**F1-score** provides a balance between precision and recall.

**ROC-AUC** measures the model's ability to distinguish between diabetic and non-diabetic cases across classification thresholds.

The notebook ranks the models according to **F1-score** and selects the highest-performing model.

## Saved Model

The trained best-performing model is saved using Joblib.

For example:

```text
XGBoost_model.pkl
```

The saved model can be loaded later without retraining:

```python
import joblib

model = joblib.load("XGBoost_model.pkl")
```

A prediction can then be generated using appropriately prepared input features:

```python
prediction = model.predict(input_data)
```

**Important:** The input data must use the same feature structure and preprocessing approach used during model training.

## Reproducibility

The project uses fixed random seeds where applicable, including:

```python
random_state=42
```

This helps produce consistent train/test splits and model results when the project is executed again under the same environment and library versions.

## Important Notes

* Keep the dataset and notebook in the expected project directory.
* Install all required Python libraries before running the notebook.
* Run notebook cells sequentially from beginning to end.
* Do not change the feature order when using the saved model.
* The `.pkl` file should be used with compatible Python and machine-learning library versions.
* This project is intended for **early-risk prediction and educational/research purposes**. It should not be treated as a substitute for professional medical diagnosis.

## Troubleshooting

### `ModuleNotFoundError: No module named 'xgboost'`

Run:

```bash
pip install xgboost
```

### `ModuleNotFoundError: No module named 'openpyxl'`

Run:

```bash
pip install openpyxl
```

### Dataset File Not Found

Check that:

```text
Diabetes Dataset.xlsx
```

is in the same directory as the notebook, or update the dataset path in the notebook.

### Model File Not Found

Check that:

```text
XGBoost_model.pkl
```

is present in the project directory before attempting to load the saved model.

## Technologies Used

* **Python**
* **Jupyter Notebook**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Microsoft Excel dataset**

## Project Objective

The primary objective is to demonstrate how machine-learning classification techniques can be applied to diabetes-related data to support **early detection and risk identification**. The project compares different algorithms and selects the most suitable model based on quantitative evaluation results.
