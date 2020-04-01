__author__ = 'Salvador Hernandez Mendoza'
__email__ = 'salvadorhm@gmail.com'
__version__ = '0.70'
import sklearn
import pandas as pd
from joblib import load
model = load('linear.joblib')
dataframe = pd.read_csv('validation.csv')
xs = dataframe[['lotsize', 'bedrooms', 'bathrms', 'stories', 'garagepl']]
predictions = model.predict(xs)
print(predictions)
