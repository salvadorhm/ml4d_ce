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
from sklearn import tree as tree_pl

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


render = web.template.render('application/views/classification', base="../master")

class ClassificationX():

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            y = webdataminingtool.classification['y']
            method = webdataminingtool.classification['method']
            
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
            return render.classification_x(method,columns,types,nulls,correlation)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self):
        try:
            try:
                filename = webdataminingtool.file['filename']
            except Exception as e:
                filename = "temp.csv"
            y = webdataminingtool.classification['y']
            method = webdataminingtool.classification['method']

            form = web.input(column = [''])
            x_cols = form.column
            webdataminingtool.classification['x']=list(x_cols)
            webdataminingtool.classification['y']= y
            dataframe = pd.read_csv(self.file)

            df_x = dataframe[x_cols]
            df_y = dataframe[y]

            x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
            
            if method == "knn":
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
                image_name = "static/images/result.png"
                plt.savefig(image_name)

                code = []
                code.append("import numpy as np")
                code.append("\n")
                code.append("from sklearn.metrics import classification_report, confusion_matrix")
                code.append("\n")
                code.append("from sklearn.model_selection import train_test_split")
                code.append("\n")
                code.append("from sklearn.neighbors import KNeighborsClassifier")
                code.append("\n")
                code.append("dataframe = pd.read_csv("+filename+")")
                code.append("\n")
                code.append("df_x = dataframe["+str(x_cols)+"]")
                code.append("\n")
                code.append("df_y = dataframe['"+y+"']")
                code.append("\n")
                code.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
                code.append("\n")
                code.append("model = KNeighborsClassifier(n_neighbors="+str(n)+")")
                code.append("\n")
                code.append("model.fit(x_train,y_train)")
                code.append("\n")
                code.append("predictions = model.predict(x_test)")
                code.append("\n")
                code.append("classification_report(y_test, predictions)")
                code.append("\n")
                code.append("confusion_matrix(y_test, predictions)")

        
                webdataminingtool.classification['filename']= filename
                webdataminingtool.classification['x']=list(x_cols)
                webdataminingtool.classification['y']= y
                webdataminingtool.classification["N_neighbors"] = n
                webdataminingtool.classification['Report'] = report
                webdataminingtool.classification['Confusion matrix'] = list(confusion)
                webdataminingtool.classification['Score'] = knn.score(x_test,y_test)
                webdataminingtool.classification['Accuracy score'] = accuracy_score(y_test, predictions)

                data_result = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
                webdataminingtool.classification['Real test values'] = list(data_result.Actual.head(10))
                webdataminingtool.classification['Predicted values'] = list(data_result.Predicted.head(10))
                webdataminingtool.classification['Python'] = "".join(code)

            elif method == "tree":
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

                figure()
                width=50
                height=24
                figure(figsize=(width,height))
                tree_pl.plot_tree(tree)
                # plt.xlabel("KNeighbors")
                # plt.ylabel("Mean error")
                # plt.title("KNeighbors test")
                image_name = "static/images/result.png"
                plt.savefig(image_name)

                code = []
                code.append("import numpy as np")
                code.append("\n")
                code.append("from sklearn.metrics import classification_report, confusion_matrix")
                code.append("\n")
                code.append("from sklearn.model_selection import train_test_split")
                code.append("\n")
                code.append("from sklearn.tree import DecisionTreeClassifier")
                code.append("\n")
                code.append("dataframe = pd.read_csv("+filename+")")
                code.append("\n")
                code.append("df_x = dataframe["+str(x_cols)+"]")
                code.append("\n")
                code.append("df_y = dataframe['"+y+"']")
                code.append("\n")
                code.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
                code.append("\n")
                code.append("model = DecisionTreeClassifier()")
                code.append("\n")
                code.append("model.fit(x_train,y_train)")
                code.append("\n")
                code.append("predictions = model.predict(x_test)")
                code.append("\n")
                code.append("classification_report(y_test, predictions)")
                code.append("\n")
                code.append("confusion_matrix(y_test, predictions)")

                webdataminingtool.classification['filename']= filename
                webdataminingtool.classification['x']=list(x_cols)
                webdataminingtool.classification['y']= y
                webdataminingtool.classification['Report'] = report
                webdataminingtool.classification['Confusion matrix'] = list(confusion)
                webdataminingtool.classification['Score'] = tree.score(x_test,y_test)
                webdataminingtool.classification['Accuracy score'] = accuracy_score(y_test, predictions)

                data_compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
                webdataminingtool.classification['Real test values'] = list(data_compare.Actual.head(10))
                webdataminingtool.classification['Predicted values'] = list(data_compare.Predicted.head(10))
                webdataminingtool.classification['Python'] = "".join(code)

            elif method == "randomf":
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

                importances = randomf.feature_importances_
                indices = np.argsort(importances)

                features = x_train.columns

                plt.figure()
                width=50
                height=24
                figure(figsize=(width,height))
                plt.title('Feature Importances')
                plt.barh(range(len(indices)), importances[indices], color='b', align='center')
                plt.yticks(range(len(indices)), features[indices])
                plt.xlabel('Relative Importance')
                image_name = "static/images/result.png"
                plt.savefig(image_name)

                code = []
                code.append("import numpy as np\n")
                code.append("\n")
                code.append("from sklearn.metrics import classification_report, confusion_matrix")
                code.append("\n")
                code.append("from sklearn.model_selection import train_test_split")
                code.append("\n")
                code.append("from sklearn.ensemble import RandomForestClassifier")
                code.append("\n")
                code.append("dataframe = pd.read_csv("+filename+")")
                code.append("\n")
                code.append("df_x = dataframe["+str(x_cols)+"]")
                code.append("\n")
                code.append("df_y = dataframe['"+y+"']")
                code.append("\n")
                code.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
                code.append("\n")
                code.append("model = RandomForestClassifier(n_estimators=80)")
                code.append("\n")
                code.append("model.fit(x_train,y_train)")
                code.append("\n")
                code.append("predictions = model.predict(x_test)")
                code.append("\n")
                code.append("classification_report(y_test, predictions)")
                code.append("\n")
                code.append("confusion_matrix(y_test, predictions)")

                webdataminingtool.classification['filename']= filename
                webdataminingtool.classification['x']=list(x_cols)
                webdataminingtool.classification['y']= y
                webdataminingtool.classification['N_estimators'] = 80
                webdataminingtool.classification['Report'] = report
                webdataminingtool.classification['Confusion matrix'] = list(confusion)
                webdataminingtool.classification['Score'] = randomf.score(x_test,y_test)
                webdataminingtool.classification['Accuracy score'] = accuracy_score(y_test, predictions)

                data_compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
                webdataminingtool.classification['Real test values'] = list(data_compare.Actual.head(10))
                webdataminingtool.classification['Predicted values'] = list(data_compare.Predicted.head(10))
                webdataminingtool.classification['Python'] = "".join(code)

            elif method == "svc":
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

                code = []
                code.append("import numpy as np")
                code.append("\n")
                code.append("from sklearn.metrics import classification_report, confusion_matrix")
                code.append("\n")
                code.append("from sklearn.model_selection import train_test_split")
                code.append("\n")
                code.append("from sklearn.svm import SVC")
                code.append("\n")
                code.append("dataframe = pd.read_csv("+filename+")")
                code.append("\n")
                code.append("df_x = dataframe["+str(x_cols)+"]")
                code.append("\n")
                code.append("df_y = dataframe['"+y+"']")
                code.append("\n")
                code.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
                code.append("\n")
                code.append("model = SVC(gamma='auto')")
                code.append("\n")
                code.append("model.fit(x_train,y_train)")
                code.append("\n")
                code.append("predictions = model.predict(x_test)")
                code.append("\n")
                code.append("classification_report(y_test, predictions)")
                code.append("\n")
                code.append("confusion_matrix(y_test, predictions)")

                webdataminingtool.classification['filename']= filename
                webdataminingtool.classification['x']=list(x_cols)
                webdataminingtool.classification['y']= y
                webdataminingtool.classification['Gamma'] = "auto"
                webdataminingtool.classification['Report'] = report
                webdataminingtool.classification['Confusion matrix'] = list(confusion)
                webdataminingtool.classification['Score'] = svc.score(x_test,y_test)
                webdataminingtool.classification['Accuracy score'] = accuracy_score(y_test, predictions)

                data_compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
                webdataminingtool.classification['Real test values'] = list(data_compare.Actual.head(10))
                webdataminingtool.classification['Predicted values'] = list(data_compare.Predicted.head(10))
                webdataminingtool.classification['Python'] = "".join(code)

            raise web.seeother('/classification_r')
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

