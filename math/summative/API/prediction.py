from fastapi import FastAPI
import numpy as np
from joblib import load
from pydantic import BaseModel


# Defining the input data model
class PredictInput(BaseModel):
    Cost_of_Living_Index: float
    Rent_Index: float
    Groceries_Index: float
    Restaurant_Price_Index: float
    Local_Purchasing_Power_Index: float

app = FastAPI()

multivariate_model = load('multivariate_model.joblib')

@app.get("/")
def read_root():
    return {"message": "Welcome to the Cost of Living Prediction API!"}

@app.post("/predict")
def predict(data: PredictInput):
    # Converting the input data to a numpy array
    input_data = np.array([
        data.Cost_of_Living_Index,
        data.Rent_Index, 
        data.Groceries_Index, 
        data.Restaurant_Price_Index, 
        data.Local_Purchasing_Power_Index
    ]).reshape(1, -1)

    # Making the prediction
    prediction = multivariate_model.predict(input_data)

    return {"prediction": float(prediction[0])}