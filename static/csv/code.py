#Librerias
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
# Creando el Dataframe para trabajar
dataframe = pd.read_csv('Leads.csv')
# Revisando si tiene NaN la columna 'TotalVisits'
dataframe['TotalVisits'].isnull().sum()
dataframe['TotalVisits'].dtypes()
dataframe['TotalVisits'].unique().tolist()
# Imputando valor a los valores NaN de la columna 'TotalVisits'
dataframe['TotalVisits'].fillna('3.445237833681204', inplace=True)
# Revisando si tiene NaN la columna 'Last Activity'
dataframe['Last Activity'].isnull().sum()
dataframe['Last Activity'].dtypes()
dataframe['Last Activity'].unique().tolist()
# Imputando valor a los valores NaN de la columna 'Last Activity'
dataframe['Last Activity'].fillna('Other', inplace=True)
# Revisando si tiene NaN la columna 'Lead Source'
dataframe['Lead Source'].isnull().sum()
dataframe['Lead Source'].dtypes()
dataframe['Lead Source'].unique().tolist()
