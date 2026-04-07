import streamlit as st

import streamlit as st
import pandas as pd

df = pd.read_csv("data/raw/Churn_Modelling.csv")

st.title("Customer Churn Intelligence Dashboard")

st.metric("Total Customers", len(df))
st.metric("Churn Rate", f"{df['Exited'].mean():.2%}")

st.subheader("Balance Distribution")
st.bar_chart(df["Balance"])