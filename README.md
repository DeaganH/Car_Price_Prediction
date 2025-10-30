# Car Price Prediction App 🚗

## Overview

This Streamlit web application predicts the selling price of a used car based on key vehicle metrics and market data. It uses a machine learning model trained on real car sales data to estimate prices in Indian Rupees (₹).

## Features

- Predicts car prices using a trained regression model.
- User-friendly interface for inputting car details.
- Supports multiple car brands, fuel types, and transmission options.
- Explains input metrics for user clarity.

## Input Metrics

- **Mileage (km/l):** Fuel efficiency.
- **Engine (cc):** Engine displacement.
- **Max Power (bhp):** Maximum power output.
- **Vehicle Age (years):** Age of the car.
- **Number of Seats:** Seating capacity.
- **Brand:** Manufacturer.
- **Fuel Type:** Petrol or diesel.
- **Transmission Type:** Manual or automatic.

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages: `streamlit`, `numpy`, `pandas`, `scikit-learn` (and any others used in your `preprocessing.py`).

### Installation

1. Clone the repository:
   ```powershell
   git clone https://github.com/DeaganH/Car_Price_Prediction.git
   cd Car_Price_Prediction
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install packages manually: `pip install streamlit numpy pandas scikit-learn`)*

3. Ensure the following files are present:
   - `app.py`
   - `preprocessing.py`
   - `random_forest_model.pkl` (or your model file)
   - Any other required files

### Running the App

Start the Streamlit app:
```powershell
streamlit run app.py
```
Open the provided local URL in your browser.

## Usage

1. Use the sidebar to navigate between "Welcome" and "Predict Price".
2. On the "Predict Price" page, enter your car details.
3. Click "Predict Price" to view the estimated selling price.

## Model

- The app loads a pre-trained regression model (`random_forest_model.pkl` or similar).
- Input data is preprocessed using custom logic in `preprocessing.py`.
---
