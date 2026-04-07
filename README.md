# Customer Lifecycle Intelligence Platform

**Production-style machine learning system simulating real-world customer management at scale**

A production-style machine learning system that predicts customer churn and recommends personalized retention strategies, inspired by real-world financial services use cases at Capital One.

---

## Business Problem

Customer churn is a critical challenge in financial services. This project:

- Identifies customers at risk of leaving  
- Recommends targeted retention strategies  
- Enables data-driven decision-making  

---

## Key Features

### Churn Prediction Model
- Classification models: Random Forest, Logistic Regression  
- Evaluation: ROC-AUC, Precision/Recall, Confusion Matrix  

### Feature Engineering
- Behavioral features (engagement score)  
- Financial ratios (balance-to-salary)  
- Risk indicators  

### Next Best Action Engine
- Personalized recommendations based on:
  - Churn probability  
  - Customer value  

### API Deployment
- Built with FastAPI  
- Real-time predictions via `/predict` endpoint  

### Model Explainability
- SHAP-based feature importance  
- Transparent decision-making  

### Monitoring
- Data drift detection  
- Model performance tracking  

### Dashboard
- Built with Streamlit  
- Visualizes churn trends and KPIs  

---

## Architecture
src/<br>
├── data/ # Data ingestion & preprocessing<br>
├── features/ # Feature engineering<br>
├── models/ # Training, evaluation, explainability<br>
├── api/ # FastAPI app<br>
├── recommender/ # Next Best Action logic<br>
├── monitoring/ # Drift detection

models/ # Saved models<br>
dashboard/ # Streamlit app<br>
data/ # Raw + processed data


---

## Tech Stack

- **Python** (pandas, scikit-learn, XGBoost)
- **API**: FastAPI  
- **Visualization**: Streamlit  
- **Explainability**: SHAP  
- **Model Storage**: joblib  

---

## How to Run

### 1. Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. Train Model

```bash
python -m src.models.train
```

### 3. Run API

```bash
uvicorn src.api.app:app --reload
```

Open: http://127.0.0.1:8000/docs

### 4. Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Example API Output

```JSON
{
  "churn_probability": 0.82,
  "churn_prediction": 1,
  "recommended_action": "Offer premium retention package"
}
```

# Future Improvements

* Deploy to AWS (S3 + Lambda/EC2)
* Add real-time streaming data pipeline
* Implement A/B testing framework

