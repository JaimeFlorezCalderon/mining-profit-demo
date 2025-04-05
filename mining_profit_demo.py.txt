
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Mining Profit Predictor", layout="centered")

st.title("Mining Company Profit Predictor")
st.write("Enter the details below to predict monthly profit:")

# Inputs del usuario
tons = st.number_input("Tons Extracted", min_value=0, value=1200)
price = st.number_input("Gold Price (USD/oz)", min_value=0.0, value=1900.0, step=10.0)
cost = st.number_input("Operating Cost (USD)", min_value=0.0, value=950000.0, step=1000.0)

# Modelo simple (coeficientes ya entrenados)
scaler = StandardScaler()
X_sample = np.array([[tons, price, cost]])
X_scaled = scaler.fit_transform(X_sample)  # Simula el escalado

# Entrenamiento simulado (mismos coeficientes del informe)
model = LinearRegression()
model.coef_ = np.array([8_590_174.59, 1_654_250.51, -126_615.52])
model.intercept_ = 70_500_947.80

# Predicción
profit = model.predict(X_scaled)[0]
st.subheader("Predicted Monthly Profit:")
st.success(f"${profit:,.2f}")

