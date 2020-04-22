__author__ = 'Salvador Hernandez Mendoza'
__email__ = 'salvadorhm@gmail.com'
__version__ = '0.75'
import sklearn
import pandas as pd
from joblib import load
model = load('randomf.joblib')
dataframe = pd.read_csv('validation.csv')
xs = dataframe[['Alcohol', 'Malic acid']]
predictions = model.predict(xs)
print(predictions)
