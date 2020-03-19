import web  # pip install web.py
import webdataminingtool
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

from sklearn.tree import DecisionTreeClassifier

from matplotlib.pyplot import figure, show
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from application.controllers.save_code import SaveCode
sc = SaveCode()


render = web.template.render('application/views/', base="master")

class TreeX():

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            y = webdataminingtool.sessions['y']

            columns = []
            types = []
            nulls = []
            correlation = []
            cols.remove(y)
            for row in cols:
                if dataframe[row].dtypes != 'object' and dataframe[y].dtype != 'object':
                    correlation.append(dataframe[y].corr(dataframe[row]))
                    types.append(dataframe[row].dtype)
                    nulls.append(dataframe[row].isnull().sum())
                    columns.append((row))
                else:
                    correlation.append(0)
                    types.append(dataframe[row].dtype)
                    nulls.append(dataframe[row].isnull().sum())
                    columns.append((row))
            return render.tree_x(columns,types,nulls,correlation)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self):
        try:
            y = webdataminingtool.sessions['y']
            form = web.input(column = [''])
            x_cols = form.column
            webdataminingtool.sessions['x']=list(x_cols)
            dataframe = pd.read_csv(self.file)

            df_x = dataframe[x_cols]
            df_y = dataframe[y]

            x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
            
            model = DecisionTreeClassifier()
            model.fit(x_train,y_train)
            predictions = model.predict(x_test)

            report = classification_report(y_test, predictions)
            confusion = confusion_matrix(y_test, predictions)

            
            webdataminingtool.sessions['Report'] = report
            webdataminingtool.sessions['Confusion matrix'] = list(confusion)
            webdataminingtool.sessions['Score'] = model.score(x_test,y_test)
            webdataminingtool.sessions['Accuracy score'] = accuracy_score(y_test, predictions)

            compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
            webdataminingtool.sessions['Real test values'] = list(compare.Actual.head(10))
            webdataminingtool.sessions['Predicted values'] = list(compare.Predicted.head(10))


            # code_lines = []
            # code_lines.append("# Preparacion del dataframe")
            # code_lines.append("df_x = dataframe["+ str(x_cols) +"]")
            # code_lines.append("df_y = dataframe['"+ y +"']")
            # code_lines.append("# x")
            # code_lines.append("df_x")
            # code_lines.append("# y")
            # code_lines.append("df_y")
            # code_lines.append("# Dataframe de entrenamiento y de prueba")
            # code_lines.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            # code_lines.append("# Model de regresion lineal")
            # code_lines.append("model = LogisticRegression()")
            # code_lines.append("# Entrenamiento del model")
            # code_lines.append("model.fit(x_train,y_train)")
            # code_lines.append("# Prueba del modelo")
            # code_lines.append("predictions = model.predict(x_test)")
            # code_lines.append("# Evaluacion del modelo")
            # code_lines.append("# Coefficients")
            # code_lines.append("classification_report(y_test, predictions)")
            # code_lines.append("# Confusion matrix")
            # code_lines.append("confusion_matrix(y_test, predictions)")
            # code_lines.append("# Score")
            # code_lines.append("model.score(x_test,y_test)")
            # code_lines.append("# Accuracy score")
            # code_lines.append("accuracy_score(y_test, predictions)")
            # code_lines.append("# Comparacion de los resultados")
            # code_lines.append("compare = pd.DataFrame({'Actual':y_test, 'Predicted':predictions})")
            # code_lines.append("# Valores de prueba")
            # code_lines.append("compare.Actual.head(10)")
            # code_lines.append("# Valores predichos")
            # code_lines.append("compare.Predicted.head(10)")
            # code_lines.append("# Bar plot")
            # code_lines.append("plt.bar(range(10),y_test.head(10))")
            # code_lines.append("plt.bar(range(10),predictions[0:10])")
            # sc.append(code_lines)
            raise web.seeother('/tree_r')
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  