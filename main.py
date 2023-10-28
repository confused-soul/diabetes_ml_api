from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from joblib import load
import json
import pandas as pd

dataset = pd.read_csv('diabetes.csv')
X = dataset.drop(columns = 'Outcome', axis = 1)
columns_to_drop = ['BP', 'ST', 'INS', 'DPF']
X = X.drop(columns=columns_to_drop)

scaler = StandardScaler()
scaler.fit(X)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):

    Pregnancies : int
    Glucose : float
    BMI : float
    Age : float

# loading the saved model
diabetes_model = load('diabetes_model.joblib')


@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters : model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    preg = input_dictionary['Pregnancies']
    glu = input_dictionary['Glucose']
    bmi = input_dictionary['BMI']
    age = input_dictionary['Age']

    input_list = scaler.transform([[preg, glu, bmi, age]])

    prediction = diabetes_model.predict(input_list)

    if prediction[0] == 0:
        return 'Not Diabetic'
    else:
        return 'Diabetic'
