import web  # pip install web.py
import app
import csv  # CSV parser
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


render = web.template.render('application/views/', base="master")

class LogisticX:

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            y = app.sessions['y']

            columns = []
            types = []
            nulls = []
            correlation = []
            cols.remove(y)
            for row in cols:
                if dataframe[row].dtypes != 'object':
                    correlation.append(dataframe[y].corr(dataframe[row]))
                    types.append(dataframe[row].dtype)
                    nulls.append(dataframe[row].isnull().sum())
                    columns.append((row))
            return render.logisticx(columns,types,nulls,correlation)
        except Exception as e:
            print(e.args)

    def POST(self):
        try:
            y = app.sessions['y']
            form = web.input(column = [''])
            x_cols = form.column
            app.sessions['x']=list(x_cols)
            dataframe = pd.read_csv(self.file)

            df_x = dataframe[x_cols]
            df_y = dataframe[y]

            x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)

            lm = LogisticRegression()
            lm.fit(x_train,y_train)

            predictions = lm.predict(x_test)

            report = classification_report(y_test, predictions)
            confusion = confusion_matrix(y_test, predictions)
            print(report)
            
            app.sessions['Report'] = report
            app.sessions['Confusion matrix'] = list(confusion)
            app.sessions['Score'] = lm.score(x_test,y_test)
            app.sessions['Accuracy score'] = accuracy_score(y_test, predictions)
            # app.sessions['Independent term'] = lm.intercept_
            # app.sessions['Mean squared error'] = mean_squared_error(y_test, predictions)
            # app.sessions['Mean absolute error'] = mean_absolute_error(y_test, predictions)
            # app.sessions['Variance'] = r2_score(y_test, predictions)

            compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
            app.sessions['Real test values'] = list(compare.Actual.head(10))
            app.sessions['Predicted values'] = list(compare.Predicted.head(10))

            raise web.seeother('/logisticr')
        except Exception as e:
            print(e.args)


  