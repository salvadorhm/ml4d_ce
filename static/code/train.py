__author__ = 'Salvador Hernandez Mendoza'
__email__ = 'salvadorhm@gmail.com'
__version__ = '0.68'
import sklearn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier
dataframe = pd.read_csv('train.csv')
df_x = dataframe[['Pclass', 'SibSp', 'Parch', 'Fare']]
df_y = dataframe['Survived']
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
model.fit(x_train,y_train)
# Dump train model to joblib
dump(model,'knn.joblib')
# Make predictions
predictions = model.predict(x_test)
classification_report(y_test, predictions)
confusion_matrix(y_test, predictions)
