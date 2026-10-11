# TELCO-CHURN-NEURAL-NETWORK# Telco Customer Churn Prediction with a Tuned Neural Network

**Name:** Zaid
**Domain:** AI/ML Internship

A deep learning project that predicts whether a telecom customer will leave (churn), using a Keras neural network that is first built as a baseline and then improved with model-tuning techniques.

## Dataset

[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers and 21 columns covering demographics, account information, subscribed services, charges, and the target `Churn` (Yes/No).

About 73.5% of customers stay and 26.5% churn, so the classes are **imbalanced**.

## Project Workflow

1. **Load, explore and clean:** shape, types and missing values; the blank `TotalCharges` entries (new customers with `tenure = 0`) are converted to numeric and set to 0; churn balance and churn vs. contract type, tenure, monthly charges and internet service are visualised.
2. **Preprocess:** binary columns mapped to 0/1, multi-class columns one-hot encoded, numeric features standardised (scaler fitted on the training set only), stratified 80/20 train/test split.
3. **Baseline model:** Keras `Sequential` network with two hidden Dense layers (32 and 16 units, ReLU) and a sigmoid output; training/validation loss curves and test accuracy.
4. **Tuned model:** L2 regularization, Batch Normalization, Dropout, Early Stopping and ReduceLROnPlateau, compared against the baseline in a before/after table and chart.
5. **Evaluation:** precision, recall, F1-score, confusion matrix and ROC curve, with a discussion of why accuracy alone is misleading on imbalanced data.
6. **Save and predict:** the best model is saved as `churn_model.keras`, and `predict_churn()` loads it and predicts for new, hypothetical customers.
7. **Short answer:** when to use a CNN versus an RNN instead of a Dense network.

## Repository Files

| File | Description |
|---|---|
| `Assignment3_Zaid.ipynb` | Complete notebook (Google Colab) with code, charts and explanations |
| `churn_model.keras` | Saved best model |
| `preprocessing.json` | Scaling values and feature order needed to prepare new customers for the model |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset |

## How to Run

1. Open `Assignment3_Zaid.ipynb` in [Google Colab](https://colab.research.google.com).
2. Choose **Runtime → Run all**.
3. When the first cell asks, upload `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Predicting for a New Customer

```python
customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5, "TotalCharges": 190.0,
}

predict_churn(customer)   # {'churn_probability': ..., 'prediction': ...}
```

## Tech Stack

Python 3, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras
