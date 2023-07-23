from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import os
from mangum import Mangum
import uvicorn
import pickle



app = FastAPI()
handler = Mangum(app)

model_path = 'models/model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)


        # Paso 2: Preprocesamiento
# Normalizar variables numéricas
mapping = {'sex': {'female': 1, 'male':0},
           'smoker': {'yes': 1, 'no':0},
           'region': {'southwest': 0, 'southeast': 0.3, 'northwest':0.6, 'northeast': 1}}
numeric_cols = ['bmi', 'age', 'children']



#-------------------------- Endpoint 1. Online prediction
class Features(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


@app.post("/online_predict")
async def predict(features_item: Features):

        #input_dict = jsonable_encoder(input_data)
        input_dict = features_item.dict()
        data = pd.DataFrame([input_dict])
        data.drop_duplicates(inplace=True)
        data.dropna(inplace=True)
        data.replace(mapping, inplace=True)
        scaler = MinMaxScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        prediction = model.predict(data)[0].item()
        return prediction


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)

