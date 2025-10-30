import streamlit as st
import numpy as np
import pandas as pd
import pickle
from preprocessing import preprocess, affordable_brands, scalar

car_brands = ['Maruti', 'Hyundai', 'Ford', 'Renault', 'Mini', 'Mercedes-Benz',
       'Toyota', 'Volkswagen', 'Honda', 'Mahindra', 'Datsun', 'Tata',
       'Kia', 'BMW', 'Audi', 'Land Rover', 'Jaguar', 'MG', 'Isuzu',
       'Porsche', 'Skoda', 'Volvo', 'Lexus', 'Jeep', 'Maserati',
       'Bentley', 'Nissan', 'ISUZU', 'Ferrari', 'Mercedes-AMG',
       'Rolls-Royce', 'Force']

st.set_page_config(page_title="Car Price Prediction App", page_icon="🚗", layout="centered")

with st.sidebar:
    st.header("Navigation")
    welcome_btn = st.button("Welcome", key="welcome_btn")
    predict_btn = st.button("Predict Price", key="predict_btn")

if 'page' not in st.session_state:
    st.session_state.page = 'Welcome'
if welcome_btn:
    st.session_state.page = 'Welcome'
if predict_btn:
    st.session_state.page = 'Predict Price'

if st.session_state.page == "Welcome":
    st.title("Welcome to the Car Price Prediction App 🚗")
    st.markdown("""
    This app predicts the selling price of a used car based on key vehicle metrics and market data.
    
    **How it works:**
    - Enter details about your car on the 'Predict Price' page.
    - The app uses a machine learning model trained on real car sales data to estimate the price.
    - All prices are predicted in **Indian Rupees (₹)**.
    
    **Input Metrics Explained:**
    - **Mileage (km/l):** Fuel efficiency of the car.
    - **Engine (cc):** Engine displacement in cubic centimeters.
    - **Max Power (bhp):** Maximum power output of the engine.
    - **Vehicle Age (years):** Age of the car in years.
    - **Number of Seats:** Seating capacity.
    - **Brand:** Manufacturer of the car.
    - **Fuel Type:** Type of fuel used (petrol or diesel).
    - **Transmission Type:** Manual or automatic gearbox.
    
    Click 'Predict Price' in the sidebar to get started!
    """)

elif st.session_state.page == "Predict Price":
    st.title("Car Price Prediction App")
    st.markdown("Enter car details below to predict the selling price.")
    col1, col2 = st.columns(2)
    with col1:
        mileage = st.number_input("Mileage (km/l)", min_value=4.0, max_value=50.0, value=4.0)
        engine = st.number_input("Engine (cc)", min_value=500, max_value=10000, value=500)
        max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=1000.0, value=30.0)
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=0)
    with col2:
        seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8, 9], index=2)
        brand = st.selectbox("Brand", car_brands)
        fuel_type = st.selectbox("Fuel Type", ["petrol", "diesel"])
        transmission_type = st.selectbox("Transmission Type", ["manual", "automatic"])
    input_data = {
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'vehicle_age': vehicle_age,
        'seats': seats,
        'brand': brand,
        'fuel_type': fuel_type,
        'transmission_type': transmission_type
    }
    if st.button("Predict Price"):
        lin_reg_model = pickle.load(open('random_forest_model.pkl', 'rb'))
        input_df = preprocess(input_data, affordable_brands, scalar)
        predicted_price = lin_reg_model.predict(input_df)[0]
        predicted_price = np.expm1(predicted_price)
        st.success(f"Predicted Selling Price: ₹{predicted_price:,.2f}")
