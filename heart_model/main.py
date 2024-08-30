from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np


# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("model.pkl")

print(model)

# Define the data model for incoming requests
class PatientData(BaseModel):
    Patient_ID: int
    Age: int
    Sex: str
    Cholesterol: int
    Blood_Pressure: str
    Heart_Rate: int
    Diabetes: int
    Alcohol_Consumption: int
    Diet: str
    Heart_Attack_Risk: int

   

# Define a prediction endpoint
@app.post("/predict/")
def predict(data: PatientData):
    # Convert the incoming data to the appropriate format for the model
    features = np.array([[data.Age, data.Sex, data.Cholesterol, data.Blood_Pressure, 
                          data.Heart_Rate, data.Diabetes, data.Alcohol_Consumption,data.Diet
                          ]])
    # Make the prediction
    prediction = model.predict(features)
    risk = "High" if prediction[0] == 1 else "Low"
    
    return {"heart_attack_risk": risk}

#Run the FastAPI app (optional if using uvicorn directly)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
