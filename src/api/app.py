from fastapi import FastAPI
import joblib
import pandas as pd

from src.recommender.next_best_action import recommend_action

app = FastAPI()

model = joblib.load("models/churn_model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    proba = model.predict_proba(df)[0][1]
    prediction = int(proba > 0.5)

    action = recommend_action(proba, data.get("Balance", 0))

    return {
        "churn_probability": float(proba),
        "churn_prediction": prediction,
        "recommended_action": action
    }

@app.get("/")
def home():
    return {"message": "Churn Model API running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    prediction = model.predict(df)[0]

    return {"churn_prediction": int(prediction)}