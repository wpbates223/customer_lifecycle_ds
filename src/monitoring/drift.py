import pandas as pd

def detect_drift(train_df, new_df, threshold=0.1):
    drift_report = {}

    for col in train_df.columns:
        if train_df[col].dtype != "object":
            train_mean = train_df[col].mean()
            new_mean = new_df[col].mean()

            drift = abs(train_mean - new_mean)

            drift_report[col] = drift > threshold

    return drift_report