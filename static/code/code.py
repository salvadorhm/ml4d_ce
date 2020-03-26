# Libraries
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.mlab as mlab
from matplotlib.pyplot import figure, show
# Loading Dataframe 
dataframe = pd.read_csv('train')
# Describe dataframe
dataframe.describe()
# Dataframe
dataframe
# KNeighbors Classifier
import csv
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
dataframe = pd.read_csv('train.csv')
df_x = dataframe[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']]
df_y = dataframe['Wine Type']
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
xs = dataframe_test[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']]
ys = dataframe_test['Wine Type']
predictions = model.predict(xs)
data_compare_test = pd.DataFrame({'Actual':ys, 'Predicted':predictions})
data_compare_test
