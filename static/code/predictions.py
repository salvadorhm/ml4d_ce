__author__ = 'Salvador Hernandez Mendoza'
__email__ = 'salvadorhm@gmail.com'
__version__ = '0.9.0'
import csv
import pandas as pd
from joblib import load
model = load('svc.joblib')
dataframe = pd.read_csv('validation.csv')
xs = dataframe[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']]
predictions = model.predict(xs)
print(predictions)
