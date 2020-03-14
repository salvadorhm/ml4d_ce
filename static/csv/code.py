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
dataframe = pd.read_csv('droop_filenames.csv')
# Descripcion del dataframe
dataframe.describe()
# Dataframe
dataframe
# Head
dataframe.head()
# Preparacion del dataframe
df_x = dataframe[['x36', 'y36', 'x37', 'y37', 'x38', 'y38', 'x39', 'y39', 'x40', 'y40', 'x41', 'y41', 'x42', 'y42', 'x43', 'y43', 'x44', 'y44', 'x45', 'y45', 'x46', 'y46', 'x47', 'y47', 'x48', 'y48', 'x49', 'y49', 'x50', 'y50', 'x51', 'y51', 'x52', 'y52', 'x53', 'y53', 'x54', 'y54', 'x55', 'y55', 'x56', 'y56', 'x57', 'y57', 'x58', 'y58', 'x59', 'y59', 'x60', 'y60', 'x61', 'y61', 'x62', 'y62', 'x63', 'y63', 'x64', 'y64', 'x65', 'y65', 'x66', 'y66', 'x67', 'y67']]
df_y = dataframe['droop']
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
Bar plot
plt.bar(range(10),y_test.head(10))
plt.bar(range(10),predictions[0:10])
# Preparacion del dataframe
df_x = dataframe[['x36', 'y36', 'x37', 'y37', 'x38', 'y38', 'x39', 'y39', 'x40', 'y40', 'x41', 'y41', 'x42', 'y42', 'x43', 'y43', 'x44', 'y44', 'x45', 'y45', 'x46', 'y46', 'x47', 'y47', 'x48', 'y48', 'x49', 'y49', 'x50', 'y50', 'x51', 'y51', 'x52', 'y52', 'x53', 'y53', 'x54', 'y54', 'x55', 'y55', 'x56', 'y56', 'x57', 'y57', 'x58', 'y58', 'x59', 'y59', 'x60', 'y60', 'x61', 'y61', 'x62', 'y62', 'x63', 'y63', 'x64', 'y64', 'x65', 'y65', 'x66', 'y66', 'x67', 'y67']]
df_y = dataframe['droop']
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
Bar plot
plt.bar(range(10),y_test.head(10))
plt.bar(range(10),predictions[0:10])
# Preparacion del dataframe
df_x = dataframe[['x36', 'y36', 'x37', 'y37', 'x38', 'y38', 'x39', 'y39', 'x40', 'y40', 'x41', 'y41', 'x42', 'y42', 'x43', 'y43', 'x44', 'y44', 'y45', 'x46', 'y46', 'x47', 'y47', 'x48', 'y48', 'x49', 'y49', 'x50', 'y50', 'x51', 'y51', 'x52', 'y52', 'x53', 'y53', 'x54', 'y54', 'x55', 'y55', 'x56', 'y56', 'x57', 'y57', 'x58', 'y58', 'x59', 'y59', 'x60', 'y60', 'x61', 'y61', 'x62', 'y62', 'x63', 'y63', 'x64', 'y64', 'x65', 'y65', 'x66', 'y66', 'x67', 'y67']]
df_y = dataframe['droop']
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
Bar plot
plt.bar(range(10),y_test.head(10))
plt.bar(range(10),predictions[0:10])
