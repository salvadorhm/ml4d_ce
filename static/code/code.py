# Libraries
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.mlab as mlab
from matplotlib.pyplot import figure, show
# Loading Dataframe 
dataframe = pd.read_csv('train.csv')
# Describe dataframe
dataframe.describe()
# Dataframe
dataframe
# Revisando si tiene NaN la columna 'room'
dataframe['room'].isnull().sum()
# Describe columna 'room'
dataframe['room'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Imputando valor a los valores NaN de la columna 'room'
dataframe['room'].fillna('0', inplace=True)
# Revisando si tiene NaN la columna 'room'
dataframe['room'].isnull().sum()
# Describe columna 'room'
dataframe['room'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'room'
dataframe['room'] = dataframe['room'].astype('int64')
