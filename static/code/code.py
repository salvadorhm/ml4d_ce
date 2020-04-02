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
# Describe
dataframe.describe()
# Describe 'mul
dataframe['mul'].describe()
# Revisando si tiene NaN la columna 'mul'
dataframe['mul'].isnull().sum()
# Describe columna 'mul'
dataframe['mul'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Drop
dataframe.drop(['mul'],axis=1,inplace=True)
# Revisando si tiene NaN la columna 'mul'
dataframe['mul'].isnull().sum()
# Describe columna 'mul'
dataframe['mul'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Describe 'x0
dataframe['x0'].describe()
# Correlation
dataframe.corr()
# Describe 'droop
dataframe['droop'].describe()
# Describe 'filename
dataframe['filename'].describe()
# Describe 'x0
dataframe['x0'].describe()
# Describe
dataframe.describe()
# Describe 'mul
dataframe['mul'].describe()
# Revisando si tiene NaN la columna 'mul'
dataframe['mul'].isnull().sum()
# Describe columna 'mul'
dataframe['mul'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Drop
dataframe.drop(['mul'],axis=1,inplace=True)
# Revisando si tiene NaN la columna 'mul'
dataframe['mul'].isnull().sum()
# Describe columna 'mul'
dataframe['mul'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'mul'
dataframe['mul'].isnull().sum()
# Describe columna 'mul'
dataframe['mul'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Revisando si tiene NaN la columna 'mul'
dataframe['mul'].isnull().sum()
# Describe columna 'mul'
dataframe['mul'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Drop
dataframe.drop(['mul'],axis=1,inplace=True)
# Describe
dataframe.describe()
# KNeighbors Classifier
import sklearn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
dataframe = pd.read_csv('train.csv')
df_x = dataframe[['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15', 'y15', 'x16', 'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20', 'y20', 'x21', 'y21', 'x22', 'y22', 'x23', 'y23', 'x24', 'y24', 'x25', 'y25', 'x26', 'y26', 'x27', 'y27', 'x28', 'y28', 'x29', 'y29', 'x30', 'y30', 'x31', 'y31', 'x32', 'y32', 'x33', 'y33', 'x34', 'y34', 'x35', 'y35', 'x36', 'y36', 'x37', 'y37', 'x38', 'y38', 'x39', 'y39', 'x40', 'y40', 'x41', 'y41', 'x42', 'y42', 'x43', 'y43', 'x44', 'y44', 'x45', 'y45', 'x46', 'y46', 'x47', 'y47', 'x48', 'y48', 'x49', 'y49', 'x50', 'y50', 'x51', 'y51', 'x52', 'y52', 'x53', 'y53', 'x54', 'y54', 'x55', 'y55', 'x56', 'y56', 'x57', 'y57', 'x58', 'y58', 'x59', 'y59', 'x60', 'y60', 'x61', 'y61', 'x62', 'y62', 'x63', 'y63', 'x64', 'y64', 'x65', 'y65', 'x66', 'y66', 'x67', 'y67']]
df_y = dataframe['droop']
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
model.fit(x_train,y_train)
dump(model,'knn.joblib')
predictions = model.predict(x_test)
# Classification report
print(classification_report(y_test, predictions))
# Confusion matrix
confusion_matrix(y_test, predictions)
# Score
model.score(x_test,y_test)
# Accuracy score
accuracy_score(y_test, predictions)
# Data compare
data_compare = pd.DataFrame({'Actual':y_test, 'Predicted':predictions})
# Compare
data_compare
# Load fit model and predict
import csv
import pandas as pd
from joblib import dump, load
model = load('knn.joblib')
dataframe_test = pd.read_csv('validation.csv')
xs = dataframe_test[['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15', 'y15', 'x16', 'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20', 'y20', 'x21', 'y21', 'x22', 'y22', 'x23', 'y23', 'x24', 'y24', 'x25', 'y25', 'x26', 'y26', 'x27', 'y27', 'x28', 'y28', 'x29', 'y29', 'x30', 'y30', 'x31', 'y31', 'x32', 'y32', 'x33', 'y33', 'x34', 'y34', 'x35', 'y35', 'x36', 'y36', 'x37', 'y37', 'x38', 'y38', 'x39', 'y39', 'x40', 'y40', 'x41', 'y41', 'x42', 'y42', 'x43', 'y43', 'x44', 'y44', 'x45', 'y45', 'x46', 'y46', 'x47', 'y47', 'x48', 'y48', 'x49', 'y49', 'x50', 'y50', 'x51', 'y51', 'x52', 'y52', 'x53', 'y53', 'x54', 'y54', 'x55', 'y55', 'x56', 'y56', 'x57', 'y57', 'x58', 'y58', 'x59', 'y59', 'x60', 'y60', 'x61', 'y61', 'x62', 'y62', 'x63', 'y63', 'x64', 'y64', 'x65', 'y65', 'x66', 'y66', 'x67', 'y67']]
ys = dataframe_test['droop']
predictions = model.predict(xs)
data_compare_test = pd.DataFrame({'Actual':ys, 'Predicted':predictions})
data_compare_test
# Revisando si tiene NaN la columna 'distancia_0'
dataframe['distancia_0'].isnull().sum()
# Describe columna 'distancia_0'
dataframe['distancia_0'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Drop
dataframe.drop(['distancia_0'],axis=1,inplace=True)
# Describe 'distancia_0
dataframe['distancia_0'].describe()
# Revisando si tiene NaN la columna 'distancia_0'
dataframe['distancia_0'].isnull().sum()
# Describe columna 'distancia_0'
dataframe['distancia_0'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Drop
dataframe.drop(['distancia_0'],axis=1,inplace=True)
# Describe 'distancia_0
dataframe['distancia_0'].describe()
# Revisando si tiene NaN la columna 'distancia_0'
dataframe['distancia_0'].isnull().sum()
# Describe columna 'distancia_0'
dataframe['distancia_0'].describe()
# Revisando si tiene NaN la columna 'static/js/control.js'
dataframe['static/js/control.js'].isnull().sum()
# Describe columna 'static/js/control.js'
dataframe['static/js/control.js'].describe()
# Revisando si tiene NaN la columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].isnull().sum()
# Describe columna 'static/icon/favicon.png'
dataframe['static/icon/favicon.png'].describe()
# Drop
dataframe.drop(['distancia_0'],axis=1,inplace=True)
# Describe 'distancia_0
dataframe['distancia_0'].describe()
