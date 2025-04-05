import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Configuración de la página
st.set_page_config(page_title="Mining Profit Predictor", layout="centered")

st.title("Mining Company Profit Predictor")
st.write("Enter the details below to predict monthly profit:")

# Crear un formulario para que solo se ejecute al presionar "Predict"
with st.form(key='prediction_form'):
    tons = st.number_input("Tons Extracted", min_value=0, value=1200)
    price = st.number_input("Gold Price (USD/oz)", min_value=0.0, value=1900.0, step=10.0)
    cost = st.number_input("Operating Cost (USD)", min_value=0.0, value=950000.0, step=1000.0)
    submit_button = st.form_submit_button(label="Predict")

# Solo calcular si se presiona el botón
if submit_button:
    # Modelo con coeficientes simples y claros (de ejemplo)
    # Puedes ajustarlos para que se parezcan a los reales si deseas
    model = LinearRegression()
    model.coef_ = np.array([3200, 1500, -1])  # toneladas, precio oro, costo operativo
    model.intercept_ = 500000  # punto de arranque base

    X = np.array([[tons, price, cost]])
    predicted_profit = model.predict(X)[0]

    # Mostrar resultado
    st.subheader("Predicted Monthly Profit:")
    st.success(f"${predicted_profit:,.2f}")
