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
# Revisando si tiene NaN la columna 'Unnamed: 0'
dataframe['Unnamed: 0'].isnull().sum()
# Describe columna 'Unnamed: 0'
dataframe['Unnamed: 0'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'Unnamed: 0'
dataframe['Unnamed: 0'] = dataframe['Unnamed: 0'].astype('str')
# Revisando si tiene NaN la columna 'Unnamed: 0'
dataframe['Unnamed: 0'].isnull().sum()
# Describe columna 'Unnamed: 0'
dataframe['Unnamed: 0'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Renombar la columna 'Unnamed: 0'
dataframe.rename(columns={'Unnamed: 0':'sin_nombre'}, inplace=True)
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('str')
# Revisando si tiene NaN la columna 'garagepl'
dataframe['garagepl'].isnull().sum()
# Describe columna 'garagepl'
dataframe['garagepl'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'garagepl'
dataframe['garagepl'] = dataframe['garagepl'].astype('str')
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('str')
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('str')
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('str')
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('str')
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('str')
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('float64')
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('str')
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('int64')
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('str')
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('str')
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('bool')
# Describe 'sin_nombre
dataframe['sin_nombre'].describe()
# Countplot
sn.countplot(data=dataframe, y='sin_nombre')
# Histogram de sin_nombre
sn.distplot(dataframe[sin_nombre])
# Revisando si tiene NaN la columna 'sin_nombre'
dataframe['sin_nombre'].isnull().sum()
# Describe columna 'sin_nombre'
dataframe['sin_nombre'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Cambiando el tipo dato de la columna 'sin_nombre'
dataframe['sin_nombre'] = dataframe['sin_nombre'].astype('int64')
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'y' por 'Y' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('y','Y'))
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'y' por 'Y' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('y','Y'))
# Describe 'driveway
dataframe['driveway'].describe()
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'y' por 'Y' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('y','Y'))
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'y' por 'H' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('y','H'))
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'y' por 'h' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('y','h'))
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'y' por 'h' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('y','h'))
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'y' por 'h' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('y','h'))
# Describe 'driveway
dataframe['driveway'].describe()
# Describe 'driveway
dataframe['driveway'].describe()
# Countplot
sn.countplot(data=dataframe, y='driveway')
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'y' por 'h' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('y','h'))
# Describe 'driveway
dataframe['driveway'].describe()
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'y' por 'h' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('y','h'))
# Describe 'driveway
dataframe['driveway'].describe()
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'y' por 'h' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('y','h'))
# Describe 'driveway
dataframe['driveway'].describe()
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'es' por 'e' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('es','e'))
# Describe 'driveway
dataframe['driveway'].describe()
# Describe columna 'driveway'
dataframe['driveway'].describe()
# Remplazando 'he' por 'yes' en la columna 'driveway'
dataframe['driveway'].apply(lambda x: x.replace('he','yes'))
# Describe 'driveway
dataframe['driveway'].describe()
# Countplot
sn.countplot(data=dataframe, y='driveway')
# Countplot
sn.countplot(data=dataframe, y='driveway')
# Describe columna 'recroom'
dataframe['recroom'].describe()
# Remplazando 'y' por 'h' en la columna 'recroom'
dataframe['recroom'].replace({'y':'h'}, inplace=True, regex=True)
# Describe 'driveway
dataframe['driveway'].describe()
