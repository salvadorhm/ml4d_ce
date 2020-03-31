__author__ = 'Salvador Hernandez Mendoza'
__email__ = 'salvadorhm@gmail.com'
__version__ = '0.68'
import sklearn
import pandas as pd
from joblib import load
model = load('knn.joblib')
dataframe = pd.read_csv('validation.csv')
xs = dataframe[['Pclass', 'SibSp', 'Parch', 'Fare']]
predictions = model.predict(xs)
print(predictions)
