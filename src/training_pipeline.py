import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Paso 1: Cargar el archivo CSV
data_path = "data/data.csv"
data = pd.read_csv(data_path)

# Paso 2: Preprocesamiento
# Normalizar variables numéricas
mapping = {'sex': {'female': 1, 'male':0},
           'smoker': {'yes': 1, 'no':0},
           'region': {'southwest': 0, 'southeast': 0.3, 'northwest':0.6, 'northeast': 1}}
numeric_cols = ['bmi', 'age', 'children']

# Eliminar duplicados
data.drop_duplicates(inplace=True)
# Eliminar valores nulos y NaN
data.dropna(inplace=True)
#mapear 
data.replace(mapping, inplace=True)
#standarizar
scaler = MinMaxScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])


# Paso 3: Separar la data de entrenamiento y prueba
X = data.drop("charges", axis=1)
y = data["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Paso 5: Evaluar el modelo entrenado
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Coeficiente de determinación (R^2) en el conjunto de entrenamiento: {train_score:.3f}")
print(f"Coeficiente de determinación (R^2) en el conjunto de prueba: {test_score:.3f}")

# Paso 6: Guardar el modelo en un archivo model.pkl en la carpeta models
models_folder = "models"
os.makedirs(models_folder, exist_ok=True)

model_path = os.path.join(models_folder, "model.pkl")
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"Modelo guardado en {model_path}")