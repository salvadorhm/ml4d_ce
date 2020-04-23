__author__ = 'Salvador Hernandez Mendoza'
__email__ = 'salvadorhm@gmail.com'
__version__ = '0.77'
import sklearn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
dataframe = pd.read_csv('train.csv')
df_x = dataframe[['Alcohol', 'Ash']]
df_y = dataframe['Wine Type']
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
model = RandomForestClassifier(n_estimators=50,criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=1, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
model.fit(x_train,y_train)
# Dump train model to joblib
dump(model,'randomf.joblib')
# Make predictions
predictions = model.predict(x_test)
classification_report(y_test, predictions)
confusion_matrix(y_test, predictions)
