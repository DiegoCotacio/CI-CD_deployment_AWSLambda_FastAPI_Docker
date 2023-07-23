from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import pandas as pd
import uvicorn
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import os
from mangum import Mangum



app = FastAPI()
handler = Mangum(app)


#---------------- LIST OF FIXED PREPROCESSING STEPS -----------------------------

def map_categorical_features(input_df):
        mapping = {
          'sex': {'female': 1, 'male':0},
          'smoker': {'yes': 1, 'no':0},
          'region': {'southwest': 0, 'southeast': 0.3, 'northwest':0.6, 'northeast': 1}}
        input_df.replace(mapping, inplace=True)
        return input_df

def normalize_numeric_features(input_df):
    input_df = input_df.copy()  # Copia el DataFrame
    numeric_cols = ['bmi', 'age', 'children']
    scaler = MinMaxScaler()
    input_df[numeric_cols] = scaler.fit_transform(input_df[numeric_cols])
    return input_df

def impute_missing_values(input_df):
    # Variables categóricas
    categorical_cols = input_df.select_dtypes(include='object').columns
    #categorical_cols = categorical_cols.drop('churn')  # Excluir 'churn'
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    input_df[categorical_cols] = categorical_imputer.fit_transform(input_df[categorical_cols])

    # Variables numéricas
    numeric_cols = input_df.select_dtypes(include=['float64', 'int64']).columns
    numeric_imputer = SimpleImputer(strategy='mean')
    input_df[numeric_cols] = numeric_imputer.fit_transform(input_df[numeric_cols])

    return input_df

def remove_duplicates(input_df):
    input_df.drop_duplicates(inplace=True)
    return input_df

def format_dtypes(input_df):
    input_df['sex'] = input_df['sex'].astype(str)
    input_df['smoker'] = input_df['smoker'].astype(str)
    input_df['region'] = input_df['region'].astype(str)
    input_df['bmi'] = input_df['bmi'].astype(float)
    input_df['age'] = input_df['age'].astype(int)
    input_df['children'] = input_df['children'].astype(int)
    return input_df

#-----------------**** MODEL ENPOINTS  *****----------------------------------

model_path = 'models/model.pkl'
# Cargar el modelo entrenado desde el archivo
model = load(model_path)




#-------------------------- Endpoint 1. Online prediction
class Features(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.post("/online_predict")
async def predict():

        #input_dict = jsonable_encoder(input_data)
        input_dict = features_item.dict()
        input_df = pd.DataFrame([input_dict])
        original_row = input_df.copy()
        input_df = format_dtypes(input_df)
        #preprocess
        input_df = normalize_numeric_features(input_df)
        input_df = map_categorical_features(input_df)
        #generate predictions
        prediction = model.predict(input_df)[0].item()
        #save monitoring input

        original_row['prediction'] = prediction
        background_tasks.add_task(save_predictions_to_bigquery, original_row)

        return prediction

