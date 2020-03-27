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
# Preparacion del dataframe
df_x = dataframe[['Pclass', 'SibSp', 'Parch', 'Fare']]
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
# Bar plot
plt.bar(range(10),y_test.head(10))
plt.bar(range(10),predictions[0:10])
# Preparacion del dataframe
df_x = dataframe[['Pclass', 'SibSp', 'Parch', 'Fare']]
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
# Bar plot
plt.bar(range(10),y_test.head(10))
plt.bar(range(10),predictions[0:10])
# Preparacion del dataframe
df_x = dataframe[['Pclass', 'SibSp', 'Parch', 'Fare']]
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
# Bar plot
plt.bar(range(10),y_test.head(10))
plt.bar(range(10),predictions[0:10])
# Preparacion del dataframe
df_x = dataframe[['Pclass', 'SibSp', 'Parch', 'Fare']]
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
# Bar plot
plt.bar(range(10),y_test.head(10))
plt.bar(range(10),predictions[0:10])
# Preparacion del dataframe
df_x = dataframe[['Pclass', 'SibSp', 'Parch', 'Fare']]
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
# Bar plot
plt.bar(range(10),y_test.head(10))
plt.bar(range(10),predictions[0:10])
