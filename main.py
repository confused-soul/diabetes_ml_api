from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from joblib import load
import json

X = [
    [6, 148, 33.6, 50],
    [1, 85, 26.6, 31],
    [8, 183, 23.3, 32],
    [1, 89, 28.1, 21],
    [0, 137, 43.1, 33],
    [5, 116, 25.6, 30],
    [3, 78, 31, 26],
    [10, 115, 35.3, 29],
    [2, 197, 30.5, 53],
    [8, 125, 0, 54],
    [4, 110, 37.6, 30],
    [10, 168, 38, 34],
    [10, 139, 27.1, 57],
    [1, 189, 30.1, 59],
    [5, 166, 25.8, 51],
    [7, 100, 30, 32],
    [0, 118, 45.8, 31],
    [7, 107, 29.6, 31],
    [1, 103, 43.3, 33],
    [1, 115, 34.6, 32],
    [3, 126, 39.3, 27],
    [8, 99, 35.4, 50],
    [7, 196, 39.8, 41],
    [9, 119, 29, 29],
    [11, 143, 36.6, 51],
    [10, 125, 31.1, 41],
    [7, 147, 39.4, 43],
    [1, 97, 23.2, 22],
    [13, 145, 22.2, 57],
    [5, 117, 34.1, 38],
    [5, 109, 36, 60],
    [3, 158, 31.6, 28],
    [3, 88, 24.8, 22],
    [6, 92, 19.9, 28],
    [10, 122, 27.6, 45],
    [4, 103, 24, 33],
    [11, 138, 33.2, 35],
    [9, 102, 32.9, 46],
    [2, 90, 38.2, 27],
    [4, 111, 37.1, 56],
    [3, 180, 34, 26],
    [7, 133, 40.2, 37],
    [7, 106, 22.7, 48],
    [9, 171, 45.4, 54],
    [7, 159, 27.4, 40],
    [0, 180, 42, 25],
    [1, 146, 29.7, 29],
    [2, 71, 28, 22],
    [7, 103, 39.1, 31],
    [7, 105, 0, 24],
    [1, 103, 19.4, 22],
    [1, 101, 24.2, 26],
    [5, 88, 24.4, 30],
    [8, 176, 33.7, 58],
    [7, 150, 34.7, 42],
    [1, 73, 23, 21],
    [7, 187, 37.7, 41],
    [0, 100, 46.8, 31],
    [0, 146, 40.5, 44],
    [0, 105, 41.5, 22],
    [2, 84, 0, 21],
    [8, 133, 32.9, 39],
    [5, 44, 25, 36],
    [2, 141, 25.4, 24],
    [7, 114, 32.8, 42],
    [5, 99, 29, 32],
    [0, 109, 32.5, 38],
    [2, 109, 42.7, 22],
    [1, 95, 28.1, 30],
    [5, 111, 30.1, 23],
    [8, 98, 27.3, 32],
    [7, 148, 29.5, 41],
    [0, 101, 31.6, 24],
    [0, 147, 34.3, 28],
    [0, 81, 46.3, 38],
    [6, 99, 36, 32],
    [4, 109, 35.5, 31]
]


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
    Glucose : int
    BMI : float
    Age : int
    

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


