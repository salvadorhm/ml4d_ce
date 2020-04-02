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
# Histogram de room
sn.distplot(dataframe[room])
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
dataframe['room'].fillna('14.9', inplace=True)
# Describe columna 'price'
dataframe['price'].describe()
# Remplazando '$' por '' en la columna 'price'
dataframe['price']=dataframe['price'].str.replace('$','')
# Describe columna 'price'
dataframe['price'].describe()
# Remplazando ',' por '' en la columna 'price'
dataframe['price']=dataframe['price'].str.replace(',','')
# Revisando si tiene NaN la columna 'price'
dataframe['price'].isnull().sum()
# Describe columna 'price'
dataframe['price'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Renombar la columna 'price'
dataframe.rename(columns={'price':'precio'}, inplace=True)
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
# Renombar la columna 'room'
dataframe.rename(columns={'room':'habitaciones'}, inplace=True)
# Revisando si tiene NaN la columna 'place'
dataframe['place'].isnull().sum()
# Describe columna 'place'
dataframe['place'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Renombar la columna 'place'
dataframe.rename(columns={'place':'lugares'}, inplace=True)
# Describe columna 'lugares'
dataframe['lugares'].describe()
# Remplazando 'center' por '1' en la columna 'lugares'
dataframe['lugares'].replace({'center':'1'}, inplace=True, regex=True)
# Describe columna 'lugares'
dataframe['lugares'].describe()
# Remplazando 'near' por '2' en la columna 'lugares'
dataframe['lugares'].replace({'near':'2'}, inplace=True, regex=True)
# Preparacion del dataframe
df_x = dataframe[['habitaciones', 'lugares']]
df_y = dataframe['precio']
# Normalizar dataframe
df_nor_x = (df_x - df_x.mean())/df_x.std()
# x
df_nor_x
# y
df_nor_y = (df_y - df_y.mean())/df_y.std()
df_nor_y
# Dataframe de entrenamiento y de prueba
x_train, x_test, y_train, y_test = train_test_split(df_nor_x,df_nor_y,test_size=0.3,random_state=42)
# Model de regresion lineal
model = LinearRegression()
# Entrenamiento del model
model.fit(x_train,y_train)
# Prueba del modelo
predictions = model.predict(x_test)
# Evaluacion del modelo
# Coefficients
model.coef_
# Independent term
model.intercept_
# Mean squared error
mean_squared_error(y_test, predictions)
# Mean absolute error
mean_absolute_error(y_test, predictions)
# Variance
r2_score(y_test, predictions)
# Comparacion de los resultados
compare = pd.DataFrame({'Actual':y_test, 'Predicted':predictions})
# Valores de prueba
compare.Actual.head(10)
# Valores predichos
compare.Predicted.head(10)
# Grafica scatter
plt.scatter(y_test,predictions)
# Grafica de distribucion
sn.distplot(y_test - predictions)
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Heatmap corr
correlation = dataframe.corr()
sn.heatmap(correlation,annot=True)
# Describe
dataframe.describe()
# Correlation
dataframe.corr()
