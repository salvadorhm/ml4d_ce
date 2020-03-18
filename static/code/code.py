# Librerias
import csv # CSV parser
import json  # json parser
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import matplotlib.mlab as mlab
from sklearn.linear_model import LogisticRegression
from matplotlib.pyplot import figure, show
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
# Creando el Dataframe para trabajar
dataframe = pd.read_csv('droop.csv')
# Descripcion del dataframe
dataframe.describe()
# Dataframe
dataframe
# Head
dataframe.head()
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Preparacion del dataframe
df_x = dataframe[['x0', 'y0', 'x1', 'y1']]
df_y = dataframe['drop']
# x
df_x
# y
df_y
# Dataframe de entrenamiento y de prueba
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
# Model de regresion lineal
model = LogisticRegression()
# Entrenamiento del model
model.fit(x_train,y_train)
# Prueba del modelo
predictions = model.predict(x_test)
# Evaluacion del modelo
# Coefficients
classification_report(y_test, predictions)
# Confusion matrix
confusion_matrix(y_test, predictions)
# Score
model.score(x_test,y_test)
# Accuracy score
accuracy_score(y_test, predictions)
# Comparacion de los resultados
compare = pd.DataFrame({'Actual':y_test, 'Predicted':predictions})
# Valores de prueba
compare.Actual.head(10)
# Valores predichos
compare.Predicted.head(10)
# Bar plot
plt.bar(range(10),y_test.head(10))
plt.bar(range(10),predictions[0:10])
# Correlation
dataframe.corr()
# Head
dataframe.head()
# Head
dataframe.head()
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Countplot
sn.countplot(x='drop', hue='x0', data= dataframe)
# Describe
dataframe.describe()
# Correlation
dataframe.corr()
# Head
dataframe.head()
# Histogram de y61
sn.distplot(dataframe[y61])
# Head
dataframe.head()
# Describe
dataframe.describe()
# Describe
dataframe.describe()
# Head
dataframe.head()
# Correlation
dataframe.corr()
# Describe
dataframe.describe()
# Describe
dataframe.describe()
# Describe
dataframe.describe()
# Head
dataframe.head()
# Head
dataframe.head()
# Head
dataframe.head()
# Correlation
dataframe.corr()
# Describe
dataframe.describe()
# Head
dataframe.head()
# Head
dataframe.head()
# Boxplot
sn.boxplot(x='drop', y='y0', data= dataframe)
# Correlation
dataframe.corr()
# Describe 'x0
dataframe['x0'].describe()
# Countplot
sn.countplot(data=dataframe, y='x0')
# Countplot
sn.countplot(x='drop', hue='x0', data= dataframe)
# Describe 'drop
dataframe['drop'].describe()
# Countplot
sn.countplot(data=dataframe, y='drop')
# Head
dataframe.head()
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Histogram de x0
sn.distplot(dataframe[x0])
# Revisando si tiene NaN la columna 'x0'
dataframe['x0'].isnull().sum()
# Describe columna 'x0'
dataframe['x0'].describe()
# Preparacion del dataframe
df_x = dataframe[['drop']]
df_y = dataframe['x0']
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
# Head
dataframe.head()
# Preparacion del dataframe
df_x = dataframe[['x0']]
df_y = dataframe['drop']
# x
df_x
# y
df_y
# Dataframe de entrenamiento y de prueba
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
# Model de regresion lineal
model = LogisticRegression()
# Entrenamiento del model
model.fit(x_train,y_train)
# Prueba del modelo
predictions = model.predict(x_test)
# Evaluacion del modelo
# Coefficients
classification_report(y_test, predictions)
# Confusion matrix
confusion_matrix(y_test, predictions)
# Score
model.score(x_test,y_test)
# Accuracy score
accuracy_score(y_test, predictions)
# Comparacion de los resultados
compare = pd.DataFrame({'Actual':y_test, 'Predicted':predictions})
# Valores de prueba
compare.Actual.head(10)
# Valores predichos
compare.Predicted.head(10)
# Bar plot
plt.bar(range(10),y_test.head(10))
plt.bar(range(10),predictions[0:10])
# Preparacion del dataframe
df_x = dataframe[['x0']]
df_y = dataframe['drop']
# x
df_x
# y
df_y
# Dataframe de entrenamiento y de prueba
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
# Model de regresion lineal
model = LogisticRegression()
# Entrenamiento del model
model.fit(x_train,y_train)
# Prueba del modelo
predictions = model.predict(x_test)
# Evaluacion del modelo
# Coefficients
classification_report(y_test, predictions)
# Confusion matrix
confusion_matrix(y_test, predictions)
# Score
model.score(x_test,y_test)
# Accuracy score
accuracy_score(y_test, predictions)
# Comparacion de los resultados
compare = pd.DataFrame({'Actual':y_test, 'Predicted':predictions})
# Valores de prueba
compare.Actual.head(10)
# Valores predichos
compare.Predicted.head(10)
# Bar plot
plt.bar(range(10),y_test.head(10))
plt.bar(range(10),predictions[0:10])
# Describe columna 'x0'
dataframe['x0'].describe()
