from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Churn Model API running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    prediction = model.predict(df)[0]

    return {"churn_prediction": int(prediction)}