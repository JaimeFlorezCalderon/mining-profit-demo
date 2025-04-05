import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Mining Profit Predictor", layout="centered")

st.title("Mining Company Profit Predictor")
st.write("Enter the details below to predict monthly profit:")

# Crear un formulario para controlar cuándo se hace la predicción
with st.form(key='prediction_form'):
    # Inputs del usuario
    tons = st.number_input("Tons Extracted", min_value=0, value=1200)
    price = st.number_input("Gold Price (USD/oz)", min_value=0.0, value=1900.0, step=10.0)
    cost = st.number_input("Operating Cost (USD)", min_value=0.0, value=950000.0, step=1000.0)
    
    # Botón de envío del formulario
    submit_button = st.form_submit_button(label="Predict")

# Solo ejecuta la predicción cuando se presiona el botón
if submit_button:
    # Simulación de escalado (como si usáramos un modelo real)
    scaler = StandardScaler()
    X_sample = np.array([[tons, price, cost]])
    X_scaled = scaler.fit_transform(X_sample)

    # Simulación del modelo (usamos coeficientes reales aprendidos)
    model = LinearRegression()
    model.coef_ = np.array([8_590_174.59, 1_654_250.51, -126_615.52])
    model.intercept_ = 70_500_947.80

    # Calcular predicción
    profit = model.predict(X_scaled)[0]

    # Mostrar resultado
    st.subheader("Predicted Monthly Profit:")
    st.success(f"${profit:,.2f}")
