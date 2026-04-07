import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.features.feature_engineering import engineer_features

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Load data
df = pd.read_csv("data/raw/Churn_Modelling.csv")
df = engineer_features(df)

y = df["Exited"]
X = df.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"])
X = pd.get_dummies(X, drop_first=True)

# Load model
model = joblib.load("models/churn_model.pkl")

# Predictions
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]