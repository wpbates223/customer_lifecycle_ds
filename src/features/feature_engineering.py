import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()

    # Financial behavior features
    df["balance_to_salary_ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

    # Engagement score
    df["engagement_score"] = (
        df["NumOfProducts"] * 2 +
        df["IsActiveMember"] * 3 -
        df["HasCrCard"]
    )

    df["high_risk"] = np.where(
        (df["CreditScore"] < 500) & (df["Balance"] > 100000),
        1, 0 
    )

    return df