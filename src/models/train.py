import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.features.feature_engineering import engineer_features

df = pd.read_csv("data/raw/Churn_Modelling.csv")

# Feature engineering
df = engineer_features(df)

# Target
y = df["Exited"]

# Drop unused columns
X = df.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"])

# One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)

feature_columns = X.columns.tolist()

import json
with open("models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/churn_model.pkl")

print("Model trained and saved successfully.")