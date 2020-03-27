__author__ = 'Salvador Hernandez Mendoza'
__email__ = 'salvadorhm@gmail.com'
__version__ = '0.9.0'
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump
dataframe = pd.read_csv('train.csv')
df_x = dataframe[['Pclass', 'SibSp', 'Parch', 'Fare']]
df_y = dataframe['Survived']
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
model = LogisticRegression()
model.fit(x_train,y_train)
# Dump train model to joblib
dump(model,'logistic.joblib')
predictions = model.predict(x_test)
print(predictions)
