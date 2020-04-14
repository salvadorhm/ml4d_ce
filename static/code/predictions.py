__author__ = 'Salvador Hernandez Mendoza'
__email__ = 'salvadorhm@gmail.com'
__version__ = '0.74'
import sklearn
import pandas as pd
from joblib import load
model = load('linear.joblib')
dataframe = pd.read_csv('validation.csv')
xs = dataframe[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
predictions = model.predict(xs)
print(predictions)
