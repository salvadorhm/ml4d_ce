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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC




from matplotlib.pyplot import figure, show

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc

from application.controllers.save_code import SaveCode
sc = SaveCode()


render = web.template.render('application/views/', base="master")

class CompareX():

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
            return render.compare_x(columns,types,nulls,correlation)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self):
        try:
            try:
                filename = webdataminingtool.file['filename']
            except Exception as e:
                filename = "temp.csv"
            y = webdataminingtool.sessions['y']
            form = web.input(column = [''])
            x_cols = form.column
            webdataminingtool.sessions['x']=list(x_cols)
            webdataminingtool.sessions['y']= y
            dataframe = pd.read_csv(self.file)

            df_x = dataframe[x_cols]
            df_y = dataframe[y]

            x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
            
            '''
            ---------------------------------------------------------------------------
            KNeighbors classifier
            ---------------------------------------------------------------------------
            '''
            tasa_error = []
            for i in range(1,30):
                knn = KNeighborsClassifier(n_neighbors=i)
                knn.fit(x_train, y_train)
                prediction_i = knn.predict(x_test)
                tasa_error.append(np.mean(prediction_i != y_test))

            min = 1
            n = 0
            for i in range(len(tasa_error)):
                if tasa_error[i] < min:
                    min = tasa_error[i]
                    n = i + 1 

            knn = KNeighborsClassifier(n_neighbors=n)
            knn.fit(x_train,y_train)
            predictions = knn.predict(x_test)

            report = classification_report(y_test, predictions)
            confusion = confusion_matrix(y_test, predictions)

            valores = range(1,30)

            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            plt.plot(valores, tasa_error, color="g", marker="o", markerfacecolor="r")
            plt.xlabel("KNeighbors")
            plt.ylabel("Mean error")
            plt.title("KNeighbors test")
            image_name = "static/images/knn.png"
            plt.savefig(image_name)
      
            webdataminingtool.knn['filename']= filename
            webdataminingtool.knn['x']=list(x_cols)
            webdataminingtool.knn['y']= y
            webdataminingtool.knn["N_neighbors"] = n
            webdataminingtool.knn['Report'] = report
            webdataminingtool.knn['Confusion matrix'] = list(confusion)
            webdataminingtool.knn['Score'] = knn.score(x_test,y_test)
            webdataminingtool.knn['Accuracy score'] = accuracy_score(y_test, predictions)

            data_result = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
            webdataminingtool.knn['Real test values'] = list(data_result.Actual.head(10))
            webdataminingtool.knn['Predicted values'] = list(data_result.Predicted.head(10))

            '''
            ---------------------------------------------------------------------------
            Decision Tree classifier
            ---------------------------------------------------------------------------
            '''
            tree = DecisionTreeClassifier()
            tree.fit(x_train,y_train)
            predictions = tree.predict(x_test)

            report = classification_report(y_test, predictions)
            confusion = confusion_matrix(y_test, predictions)

            webdataminingtool.tree['filename']= filename
            webdataminingtool.tree['x']=list(x_cols)
            webdataminingtool.tree['y']= y
            webdataminingtool.tree['Report'] = report
            webdataminingtool.tree['Confusion matrix'] = list(confusion)
            webdataminingtool.tree['Score'] = tree.score(x_test,y_test)
            webdataminingtool.tree['Accuracy score'] = accuracy_score(y_test, predictions)

            data_compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
            webdataminingtool.tree['Real test values'] = list(data_compare.Actual.head(10))
            webdataminingtool.tree['Predicted values'] = list(data_compare.Predicted.head(10))

            '''
            ---------------------------------------------------------------------------
            RandomForest Classifier
            ---------------------------------------------------------------------------
            '''
            randomf = RandomForestClassifier(n_estimators=80)
            randomf.fit(x_train,y_train)
            predictions = randomf.predict(x_test)

            report = classification_report(y_test, predictions)
            confusion = confusion_matrix(y_test, predictions)

            webdataminingtool.randomf['filename']= filename
            webdataminingtool.randomf['x']=list(x_cols)
            webdataminingtool.randomf['y']= y
            webdataminingtool.randomf['N_estimators'] = 80
            webdataminingtool.randomf['Report'] = report
            webdataminingtool.randomf['Confusion matrix'] = list(confusion)
            webdataminingtool.randomf['Score'] = randomf.score(x_test,y_test)
            webdataminingtool.randomf['Accuracy score'] = accuracy_score(y_test, predictions)

            data_compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
            webdataminingtool.randomf['Real test values'] = list(data_compare.Actual.head(10))
            webdataminingtool.randomf['Predicted values'] = list(data_compare.Predicted.head(10))

            '''
            ---------------------------------------------------------------------------
            Suport Vector MAchine Classifier
            ---------------------------------------------------------------------------
            '''
            svc = SVC(gamma='auto')
            svc.fit(x_train,y_train)
            predictions = svc.predict(x_test)

            report = classification_report(y_test, predictions)
            confusion = confusion_matrix(y_test, predictions)

            webdataminingtool.svc['filename']= filename
            webdataminingtool.svc['x']=list(x_cols)
            webdataminingtool.svc['y']= y
            webdataminingtool.svc['Gamma'] = "auto"
            webdataminingtool.svc['Report'] = report
            webdataminingtool.svc['Confusion matrix'] = list(confusion)
            webdataminingtool.svc['Score'] = svc.score(x_test,y_test)
            webdataminingtool.svc['Accuracy score'] = accuracy_score(y_test, predictions)

            data_compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
            webdataminingtool.svc['Real test values'] = list(data_compare.Actual.head(10))
            webdataminingtool.svc['Predicted values'] = list(data_compare.Predicted.head(10))

            raise web.seeother('/compare_r')
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])


  