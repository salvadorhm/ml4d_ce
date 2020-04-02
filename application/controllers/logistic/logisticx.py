import web  # pip install web.py
import ml4d
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
from joblib import dump, load
from application.controllers.save_code import SaveCode
sc = SaveCode()


render = web.template.render('application/views/logistic', base="../master")

class LogisticX:

    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            y = ml4d.sessions['y']

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
            return render.error(e.args[0])

    def POST(self):
        try:
            try:
                filename = ml4d.file['filename']
            except Exception as e:
                filename = "train.csv"
            y = ml4d.sessions['y']
            form = web.input(column = [''])
            x_cols = form.column
            ml4d.sessions['x']=list(x_cols)
            dataframe = pd.read_csv(self.file)

            df_x = dataframe[x_cols]
            df_y = dataframe[y]

            x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)

            model = LogisticRegression()
            model.fit(x_train,y_train)
            dump(model, "static/models/logistic.joblib") 


            predictions = model.predict(x_test)

            report = classification_report(y_test, predictions)
            confusion = confusion_matrix(y_test, predictions)

            compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})


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


            code = []
            code.append("import sklearn")
            code.append("\n")
            code.append("import pandas as pd")
            code.append("\n")
            code.append("from sklearn.model_selection import train_test_split")
            code.append("\n")
            code.append("from sklearn.linear_model import LogisticRegression")
            code.append("\n")
            code.append("dataframe = pd.read_csv("+filename+")")
            code.append("\n")
            code.append("df_x = dataframe["+str(x_cols)+"]")
            code.append("\n")
            code.append("df_y = dataframe['"+y+"']")
            code.append("\n")
            code.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            code.append("\n")
            code.append("model = LogisticRegression()")
            code.append("\n")
            code.append("model.fit(x_train,y_train)")
            code.append("\n")
            code.append("# Dump train model to joblib")
            code.append("\n")
            code.append("dump(model,'logistic.joblib')")
            code.append("\n")
            code.append("predictions = model.predict(x_test)")
            code.append("\n")
            code.append("print(predictions)")

            train_py = []
            train_py.append("__author__ = 'Salvador Hernandez Mendoza'")
            train_py.append("__email__ = 'salvadorhm@gmail.com'")
            train_py.append("__version__ = '"+ml4d.app_version+"'")
            train_py.append("import sklearn")
            train_py.append("import pandas as pd")
            train_py.append("from sklearn.model_selection import train_test_split")
            train_py.append("from sklearn.linear_model import LogisticRegression")
            train_py.append("from joblib import dump")
            train_py.append("dataframe = pd.read_csv('train.csv')")
            train_py.append("df_x = dataframe["+str(x_cols)+"]")
            train_py.append("df_y = dataframe['"+y+"']")
            train_py.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            train_py.append("model = LogisticRegression()")
            train_py.append("model.fit(x_train,y_train)")
            train_py.append("# Dump train model to joblib")
            train_py.append("dump(model,'logistic.joblib')")
            train_py.append("predictions = model.predict(x_test)")
            train_py.append("print(predictions)")
            sc.createCode("train.py",train_py)

            test = []
            test.append("# Load model and predict")
            test.append("\n")
            test.append("import sklearn")
            test.append("\n")
            test.append("import pandas as pd")
            test.append("\n")
            test.append("from joblib import load")
            test.append("\n")
            test.append("# Load trained model")
            test.append("\n")
            test.append("model = load('logistic.joblib')")
            test.append("\n")
            test.append("dataframe = pd.read_csv('validation.csv')")
            test.append("\n")
            test.append("xs = dataframe["+str(x_cols)+"]")
            test.append("\n")
            test.append("predictions = model.predict(xs)")
            test.append("\n")
            test.append("print(predictions)")


            predictions_py = []
            predictions_py.append("__author__ = 'Salvador Hernandez Mendoza'")
            predictions_py.append("__email__ = 'salvadorhm@gmail.com'")
            predictions_py.append("__version__ = '"+ml4d.app_version+"'")
            predictions_py.append("import sklearn")
            predictions_py.append("import pandas as pd")
            predictions_py.append("from joblib import load")
            predictions_py.append("model = load('logistic.joblib')")
            predictions_py.append("dataframe = pd.read_csv('validation.csv')")
            predictions_py.append("xs = dataframe["+str(x_cols)+"]")
            predictions_py.append("predictions = model.predict(xs)")
            predictions_py.append("print(predictions)")
            sc.createCode("predictions.py",predictions_py)

            ml4d.sessions['filename']= filename
            ml4d.sessions['y'] = y 
            ml4d.sessions['x'] = list(x_cols)
            ml4d.sessions['Report'] = report
            ml4d.sessions['Confusion matrix'] = list(confusion)
            ml4d.sessions['Score'] = model.score(x_test,y_test)
            ml4d.sessions['Accuracy score'] = accuracy_score(y_test, predictions)
            ml4d.sessions['Real test values'] = list(compare.Actual.head(10))
            ml4d.sessions['Predicted values'] = list(compare.Predicted.head(10))
            ml4d.sessions['Python'] = "".join(code)
            ml4d.sessions['Python validation'] = "".join(test)
            ml4d.sessions['Model'] = "logistic.joblib"
            ml4d.sessions['train.csv'] = "train.csv"
            ml4d.sessions['validation.csv'] = "validation.csv"
            ml4d.sessions['train.py'] = "train.py"
            ml4d.sessions['predictions.py'] = "predictions.py"


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
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])


  