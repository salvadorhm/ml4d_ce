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
# Heatmap corr
correlation = dataframe.corr()
sn.heatmap(correlation,annot=True)
# Histogram de price
sn.distplot(dataframe[price])
# Histogram de lotsize
sn.distplot(dataframe[lotsize])
# Histogram de stories
sn.distplot(dataframe[stories])
# Heatmap corr
correlation = dataframe.corr()
sn.heatmap(correlation,annot=True)
# Histogram de price
sn.distplot(dataframe[price])
# Preparacion del dataframe
df_x = dataframe[['lotsize', 'bedrooms', 'bathrms', 'stories', 'garagepl']]
df_y = dataframe['price']
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
# Histogram de price
sn.distplot(dataframe[price])
# Histogram de price
sn.distplot(dataframe[price])
# Histogram de price
sn.distplot(dataframe[price])
# Histogram de lotsize
sn.distplot(dataframe[lotsize])
# Histogram de price
sn.distplot(dataframe[price])
# Histogram de bathrms
sn.distplot(dataframe[bathrms])
# Countplot
sn.countplot(data=dataframe, y='lotsize')
# Histogram de bathrms
sn.distplot(dataframe[bathrms])
# Histogram de price
sn.distplot(dataframe[price])
# Histogram de stories
sn.distplot(dataframe[stories])
# Histogram de garagepl
sn.distplot(dataframe[garagepl])
# Countplot
sn.countplot(data=dataframe, y='recroom')
# Countplot
sn.countplot(data=dataframe, y='airco')
# Countplot
sn.countplot(data=dataframe, y='driveway')
# Preparacion del dataframe
df_x = dataframe[['lotsize', 'bedrooms', 'bathrms', 'stories', 'garagepl']]
df_y = dataframe['price']
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
# Describe
dataframe.describe()
# Preparacion del dataframe
df_x = dataframe[['lotsize', 'bedrooms', 'bathrms', 'stories', 'garagepl']]
df_y = dataframe['price']
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
# Histogram de lotsize
sn.distplot(dataframe[lotsize])
# Describe
dataframe.describe()
# Getdummies airco
dataframe = pd.get_dummies(dataframe, columns =['airco'])
# Describe 'airco_no
dataframe['airco_no'].describe()
# Histogram de airco_no
sn.distplot(dataframe[airco_no])
# Describe 'airco_yes
dataframe['airco_yes'].describe()
# Histogram de airco_yes
sn.distplot(dataframe[airco_yes])
# Describe
dataframe.describe()
# Correlation
dataframe.corr()
# Correlation
dataframe.corr()
# Correlation
dataframe.corr()
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Countplot
sn.countplot(x='bedrooms', hue='price', data= dataframe)
# Heatmap corr
correlation = dataframe.corr()
sn.heatmap(correlation,annot=True)
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Revisando si tiene NaN la columna 'Unnamed: 0'
dataframe['Unnamed: 0'].isnull().sum()
# Describe columna 'Unnamed: 0'
dataframe['Unnamed: 0'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'Unnamed: 0'
dataframe['Unnamed: 0'].isnull().sum()
# Describe columna 'Unnamed: 0'
dataframe['Unnamed: 0'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'price'
dataframe['price'].isnull().sum()
# Describe columna 'price'
dataframe['price'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'price'
dataframe['price'].isnull().sum()
# Describe columna 'price'
dataframe['price'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'Unnamed: 0'
dataframe['Unnamed: 0'].isnull().sum()
# Describe columna 'Unnamed: 0'
dataframe['Unnamed: 0'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'Unnamed: 0'
dataframe['Unnamed: 0'].isnull().sum()
# Describe columna 'Unnamed: 0'
dataframe['Unnamed: 0'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'Unnamed: 0'
dataframe['Unnamed: 0'].isnull().sum()
# Describe columna 'Unnamed: 0'
dataframe['Unnamed: 0'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'Unnamed: 0'
dataframe['Unnamed: 0'].isnull().sum()
# Describe columna 'Unnamed: 0'
dataframe['Unnamed: 0'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Describe columna 'Unnamed: 0'
dataframe['Unnamed: 0'].describe()
# Describe columna 'lotsize'
dataframe['lotsize'].describe()
# Revisando si tiene NaN la columna 'lotsize'
dataframe['lotsize'].isnull().sum()
# Describe columna 'lotsize'
dataframe['lotsize'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'bedrooms'
dataframe['bedrooms'].isnull().sum()
# Describe columna 'bedrooms'
dataframe['bedrooms'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
