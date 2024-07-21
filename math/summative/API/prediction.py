import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from fastapi import FastAPI, Query
import joblib

app = FastAPI(title="Cost of Living Prediction API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Cost of Living Prediction API!"}


multivariate_model = joblib.load('multivariate_model.joblib')

@app.post("/predict")
def predict(
    cost_of_living_index: float = Query(..., description="Cost of Living Index"),
    rent_index: float = Query(..., description="Rent Index"),
    groceries_index: float = Query(..., description="Groceries Index"),
    restaurant_price_index: float = Query(..., description="Restaurant Price Index"),
    local_purchasing_power_index: float = Query(..., description="Local Purchasing Power Index")
    
):
    input_data = np.array([cost_of_living_index, rent_index, groceries_index, restaurant_price_index, local_purchasing_power_index]).reshape(1, -1)
    prediction = multivariate_model.predict(input_data)[0]
    return {"prediction": prediction}