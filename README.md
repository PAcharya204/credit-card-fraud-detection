# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions.

The project compares different classification models and studies the effect of SMOTE on fraud detection performance. The final model is deployed as a Streamlit web application using Render.

Live Demo:
https://credit-card-fraud-detection-3k8r.onrender.com


## Project Overview

Credit card fraud detection is a classification problem where the goal is to identify whether a transaction is genuine or fraudulent.

The dataset contains transaction-related features such as distance from home, distance from the previous transaction, purchase price compared with the median purchase price, whether the transaction was made online, and whether a chip or PIN was used.

The project focuses on building and evaluating multiple machine learning models and selecting the model that performs best for detecting fraudulent transactions.


## Dataset

The dataset contains 1,000,000 transactions with 7 input features and one target variable.

The input features used in the project are:

- distance_from_home
- distance_from_last_transaction
- ratio_to_median_purchase_price
- repeat_retailer
- used_chip
- used_pin_number
- online_order

The target variable indicates whether a transaction is fraudulent or genuine.


## Data Preprocessing

The dataset was checked for missing values and duplicate records before model training.

The input features were separated from the target variable and the data was divided into training and testing sets.

Feature scaling was performed using StandardScaler.

SMOTE was also applied to the training data to study whether balancing the classes would improve fraud detection performance.


## Machine Learning Models

The following classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Each model was evaluated both before and after applying SMOTE.


## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Recall was given particular attention because missing a fraudulent transaction can be more important than incorrectly flagging a genuine transaction.


## Model Comparison

### Before SMOTE

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 95.941% | 89.643% | 60.557% | 72.284% |
| Decision Tree | 99.9985% | 99.9943% | 99.9886% | 99.9914% |
| Random Forest | 99.997% | 100.000% | 99.9657% | 99.9828% |
| XGBoost | 99.8185% | 98.8360% | 99.0904% | 98.9631% |

### After SMOTE

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 93.480% | 57.735% | 94.8115% | 71.7676% |
| Decision Tree | 99.995% | 99.9600% | 99.9828% | 99.9714% |
| Random Forest | 99.997% | 99.9886% | 99.9771% | 99.9828% |
| XGBoost | 99.8265% | 98.4559% | 99.5767% | 99.0131% |


## Final Model

The Decision Tree without SMOTE was selected as the final model.

It achieved:

- Accuracy: 99.9985%
- Precision: 99.9943%
- Recall: 99.9886%
- F1-score: 99.9914%

The model correctly detected 17,479 fraudulent transactions and missed only 2 fraudulent transactions in the test set.

Although SMOTE improved the recall of Logistic Regression and slightly improved XGBoost, it did not provide a meaningful improvement for the tree-based models.


## Feature Importance

The model was also used to examine the importance of the input features.

The feature importance values were:

- ratio_to_median_purchase_price: 42.26%
- online_order: 25.75%
- distance_from_home: 10.94%
- used_pin_number: 10.48%
- used_chip: 6.78%
- distance_from_last_transaction: 3.48%
- repeat_retailer: 0.31%

The ratio of the transaction price to the typical purchase price was the most important feature in the model.


## Streamlit Application

The trained model and scaler were saved using Joblib and used in a Streamlit application.

The application allows a user to enter transaction details and receive a prediction indicating whether the transaction is genuine or fraudulent.

The application is deployed on Render and can be accessed using the link below.

Live Demo:
https://credit-card-fraud-detection-3k8r.onrender.com


## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- SMOTE
- Joblib
- Streamlit
- Render
- Google Colab


## Project Structure

```text
credit-card-fraud-detection/
│
├── Colab Notebook/
│   └── CreditCard_Fraud_Detection.ipynb
│
└── fraud_detection_deploy/
    ├── app.py
    ├── fraud_detection_model.pkl
    ├── scaler.pkl
    └── requirements.txt
