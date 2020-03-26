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
dataframe = pd.read_csv('demo3x.csv')
# Descripcion del dataframe
dataframe.describe()
# Dataframe
dataframe
# Histogram de x
sn.distplot(dataframe[x])
# Histogram de y
sn.distplot(dataframe[y])
# Countplot
sn.countplot(data=dataframe, y='y')
# Preparacion del dataframe
df_x = dataframe[['x']]
df_y = dataframe['y']
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
