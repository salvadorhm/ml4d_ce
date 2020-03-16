import web  # pip install web.py
import wdmt
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
from sklearn.metrics import roc_curve, auc
from application.controllers.save_code import SaveCode
sc = SaveCode()


render = web.template.render('application/views/', base="master")

class LogisticX:

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            y = wdmt.sessions['y']

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
        # try:
            y = wdmt.sessions['y']
            form = web.input(column = [''])
            x_cols = form.column
            wdmt.sessions['x']=list(x_cols)
            dataframe = pd.read_csv(self.file)

            df_x = dataframe[x_cols]
            df_y = dataframe[y]

            x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)

            model = LogisticRegression()
            model.fit(x_train,y_train)

            predictions = model.predict(x_test)

            report = classification_report(y_test, predictions)
            confusion = confusion_matrix(y_test, predictions)

            print(report)
            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            plt.bar(range(10),y_test.head(10))
            plt.bar(range(10),predictions[0:10])
            
            image_name = "static/images/logistic.png"
            plt.savefig(image_name)


            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            fpr, tpr, thresholds = roc_curve(y_test, predictions)
            plt.plot(fpr,tpr)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.title('ROC curve del modelo de Droop')
            plt.xlabel('False positive rate (1-Specificity)')
            plt.ylabel('True positive rate (Sensitivity)')
            plt.grid(True)
            image_name = "static/images/roc.png"
            plt.savefig(image_name)

            
            wdmt.sessions['Report'] = report
            wdmt.sessions['Confusion matrix'] = list(confusion)
            wdmt.sessions['Score'] = model.score(x_test,y_test)
            wdmt.sessions['Accuracy score'] = accuracy_score(y_test, predictions)

            compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
            wdmt.sessions['Real test values'] = list(compare.Actual.head(10))
            wdmt.sessions['Predicted values'] = list(compare.Predicted.head(10))


            code_lines = []
            code_lines.append("# Preparacion del dataframe")
            code_lines.append("df_x = dataframe["+ str(x_cols) +"]")
            code_lines.append("df_y = dataframe['"+ y +"']")
            code_lines.append("# x")
            code_lines.append("df_x")
            code_lines.append("# y")
            code_lines.append("df_y")
            code_lines.append("# Dataframe de entrenamiento y de prueba")
            code_lines.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            code_lines.append("# Model de regresion lineal")
            code_lines.append("model = LogisticRegression()")
            code_lines.append("# Entrenamiento del model")
            code_lines.append("model.fit(x_train,y_train)")
            code_lines.append("# Prueba del modelo")
            code_lines.append("predictions = model.predict(x_test)")
            code_lines.append("# Evaluacion del modelo")
            code_lines.append("# Coefficients")
            code_lines.append("classification_report(y_test, predictions)")
            code_lines.append("# Confusion matrix")
            code_lines.append("confusion_matrix(y_test, predictions)")
            code_lines.append("# Score")
            code_lines.append("model.score(x_test,y_test)")
            code_lines.append("# Accuracy score")
            code_lines.append("accuracy_score(y_test, predictions)")
            code_lines.append("# Comparacion de los resultados")
            code_lines.append("compare = pd.DataFrame({'Actual':y_test, 'Predicted':predictions})")
            code_lines.append("# Valores de prueba")
            code_lines.append("compare.Actual.head(10)")
            code_lines.append("# Valores predichos")
            code_lines.append("compare.Predicted.head(10)")
            code_lines.append("# Bar plot")
            code_lines.append("plt.bar(range(10),y_test.head(10))")
            code_lines.append("plt.bar(range(10),predictions[0:10])")
            sc.append(code_lines)
            raise web.seeother('/logisticr')
        # except Exception as e:
            # print(e.args)


  