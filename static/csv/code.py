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
dataframe = pd.read_csv('regresion_logistica.csv')
# Descripcion del dataframe
dataframe.describe()
# Dataframe
dataframe
# Head
dataframe.head()
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Revisando si tiene NaN la columna 'Age'
dataframe['Age'].isnull().sum()
# Describe columna 'Age'
dataframe['Age'].describe()
# Imputando valor a los valores NaN de la columna 'Age'
dataframe['Age'].fillna('28', inplace=True)
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Getdummies Sex
dataframe = pd.get_dummies(dataframe, columns =[Sex])
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Revisando si tiene NaN la columna 'Embarked'
dataframe['Embarked'].isnull().sum()
# Describe columna 'Embarked'
dataframe['Embarked'].describe()
# Imputando valor a los valores NaN de la columna 'Embarked'
dataframe['Embarked'].fillna('S', inplace=True)
# Getdummies Embarked
dataframe = pd.get_dummies(dataframe, columns =[Embarked])
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Boxplot
sn.boxplot(x='Pclass', y='Age', data= dataframe)
# Preparacion del dataframe
df_x = dataframe[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
df_y = dataframe['Survived']
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
# Head
dataframe.head()
# Preparacion del dataframe
df_x = dataframe[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
df_y = dataframe['Survived']
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
df_x = dataframe[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
df_y = dataframe['Survived']
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
df_x = dataframe[['Age', 'Sex_female', 'Sex_male']]
df_y = dataframe['Survived']
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
