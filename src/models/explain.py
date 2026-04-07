import shap
import pandas as pd
import joblib
import json

from src.features.feature_engineering import engineer_features

# Load data
df = pd.read_csv("data/raw/Churn_Modelling.csv")
df = engineer_features(df)

X = df.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"])
X = pd.get_dummies(X, drop_first=True)

with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

X = X.reindex(columns=feature_columns, fill_value=0)

# Load model
model = joblib.load("models/churn_model.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Fix for binary classification (sklearn RF)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Summary plot
shap.summary_plot(shap_values, X)

