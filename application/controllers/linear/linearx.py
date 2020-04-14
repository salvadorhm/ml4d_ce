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
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import figure, show
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from joblib import dump, load
from application.controllers.save_code import SaveCode

import plotly.express as px

sc = SaveCode()

render = web.template.render('application/views/linear/', base="../master")

class LinearX:

    file = 'static/csv/train.csv'  # define el archivo donde se amodelacenan los datos

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
            return render.linearx(columns,types,nulls,correlation)
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
            # columns = form.column
            x_cols = form.column
            ml4d.sessions['x']=list(x_cols)
            dataframe = pd.read_csv(self.file)
            # columns.append(y)
            # df = dataframe[columns]

            df_x = dataframe[x_cols]
            df_y = dataframe[y]

            df_nor_x = (df_x - df_x.mean())/df_x.std()
            df_nor_y = (df_y - df_y.mean())/df_y.std()
            

            # x_train, x_test, y_train, y_test = train_test_split(df_nor_x,df_nor_y,test_size=0.3,random_state=42)
            
            x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)

            model = LinearRegression()
            model.fit(x_train,y_train)
            dump(model, "static/models/linear.joblib") 

            predictions = model.predict(x_test)
            compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
            

            df = y_test - predictions
            fig = px.histogram(df, x=ml4d.sessions['y'])
            fig.write_html("static/images/histogram.html")

            if len(list(x_test)) == 1:
                # print("graficar")
                # figure()
                # width=20
                # height=8
                # figure(figsize=(width,height))
                # # TODO grafica lineal simple, una sola x
                # ax = plt.scatter(x_test, y_test)
                # plt.plot(x_test,predictions,color='red')
                # image_name = "static/images/lineal.png"
                # ax.figure.savefig(image_name)
                # # fig = ax.get_figure()
                # #plt.close(fig)
                xs = []
                ys = []
                ps = []
                print(ml4d.sessions['x'])
                for i in x_test[ml4d.sessions['x'][0]]:
                    xs.append(i)
                for i in y_test:
                    ys.append(i)
                for i in predictions:
                    ps.append(i)

                fig = px.scatter(x=xs, y=ys)
                fig.write_html("static/images/linear.html")
            else:
                x = []
                n = 0
                for i in y_test:
                    x.append(n)
                    n+=1
                figure()
                width=20
                height=8
                figure(figsize=(width,height))
                # TODO grafica lineal simple, una sola x
                ax = plt.scatter(x, y_test)
                plt.scatter(x,predictions,color='red')
                image_name = "static/images/lineal.png"
                ax.figure.savefig(image_name)
                # fig = ax.get_figure()
                #plt.close(fig)


            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            nor = sn.distplot(y_test - predictions)
            image_name = "static/images/histogram.png"
            nor.figure.savefig(image_name)
            # fig = nor.get_figure()
            #plt.close(fig)

            code = []
            code.append("import sklearn")
            code.append("\n")
            code.append("import pandas as pd")
            code.append("\n")
            code.append("from sklearn.model_selection import train_test_split")
            code.append("\n")
            code.append("from sklearn.linear_model import LinearRegression")
            code.append("\n")
            code.append("dataframe = pd.read_csv("+filename+")")
            code.append("\n")
            code.append("df_x = dataframe["+str(x_cols)+"]")
            code.append("\n")
            code.append("df_y = dataframe['"+y+"']")
            code.append("\n")
            code.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            code.append("\n")
            code.append("model = LinearRegression()")
            code.append("\n")
            code.append("model.fit(x_train,y_train)")
            code.append("\n")
            code.append("# Dump train model to joblib")
            code.append("\n")
            code.append("dump(model,'linear.joblib')")
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
            train_py.append("from sklearn.linear_model import LinearRegression")
            train_py.append("from joblib import dump")
            train_py.append("dataframe = pd.read_csv('train.csv')")
            train_py.append("df_x = dataframe["+str(x_cols)+"]")
            train_py.append("df_y = dataframe['"+y+"']")
            train_py.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            train_py.append("model = LinearRegression()")
            train_py.append("model.fit(x_train,y_train)")
            train_py.append("# Dump train model to joblib")
            train_py.append("dump(model,'linear.joblib')")
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
            test.append("model = load('linear.joblib')")
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
            predictions_py.append("model = load('linear.joblib')")
            predictions_py.append("dataframe = pd.read_csv('validation.csv')")
            predictions_py.append("xs = dataframe["+str(x_cols)+"]")
            predictions_py.append("predictions = model.predict(xs)")
            predictions_py.append("print(predictions)")
            sc.createCode("predictions.py",predictions_py)

            ml4d.sessions['filename']= filename
            ml4d.sessions['y'] = y 
            ml4d.sessions['x'] = list(x_cols)
            ml4d.sessions['Coefficients'] = str(model.coef_)
            ml4d.sessions['Independent term'] = model.intercept_
            ml4d.sessions['Mean squared error'] = mean_squared_error(y_test, predictions)
            ml4d.sessions['Mean absolute error'] = mean_absolute_error(y_test, predictions)
            ml4d.sessions['Variance'] = r2_score(y_test, predictions)
            ml4d.sessions['Actual test values'] = list(compare.Actual.head())
            ml4d.sessions['Predicted values'] = list(compare.Predicted.head())
            ml4d.sessions['Python train'] = "".join(code)
            ml4d.sessions['Python predictions'] = "".join(test)
            ml4d.sessions['Model'] = "linear.joblib"
            ml4d.sessions['train.csv'] = "train.csv"
            # ml4d.sessions['validation.csv'] = "validation.csv"
            ml4d.sessions['train.py'] = "train.py"
            ml4d.sessions['predictions.py'] = "predictions.py"


            code_lines = []
            code_lines.append("# Preparacion del dataframe")
            code_lines.append("df_x = dataframe["+ str(x_cols) +"]")
            code_lines.append("df_y = dataframe['"+ y +"']")
            code_lines.append("# Normalizar dataframe")
            code_lines.append("df_nor_x = (df_x - df_x.mean())/df_x.std()")
            code_lines.append("# x")
            code_lines.append("df_nor_x")
            code_lines.append("# y")
            code_lines.append("df_nor_y = (df_y - df_y.mean())/df_y.std()")
            code_lines.append("df_nor_y")
            code_lines.append("# Dataframe de entrenamiento y de prueba")
            code_lines.append("x_train, x_test, y_train, y_test = train_test_split(df_nor_x,df_nor_y,test_size=0.3,random_state=42)")
            code_lines.append("# Model de regresion lineal")
            code_lines.append("model = LinearRegression()")
            code_lines.append("# Entrenamiento del model")
            code_lines.append("model.fit(x_train,y_train)")
            code_lines.append("# Prueba del modelo")
            code_lines.append("predictions = model.predict(x_test)")
            code_lines.append("# Evaluacion del modelo")
            code_lines.append("# Coefficients")
            code_lines.append("model.coef_")
            code_lines.append("# Independent term")
            code_lines.append("model.intercept_")
            code_lines.append("# Mean squared error")
            code_lines.append("mean_squared_error(y_test, predictions)")
            code_lines.append("# Mean absolute error")
            code_lines.append("mean_absolute_error(y_test, predictions)")
            code_lines.append("# Variance")
            code_lines.append("r2_score(y_test, predictions)")
            code_lines.append("# Comparacion de los resultados")
            code_lines.append("compare = pd.DataFrame({'Actual':y_test, 'Predicted':predictions})")
            code_lines.append("# Valores de prueba")
            code_lines.append("compare.Actual.head(10)")
            code_lines.append("# Valores predichos")
            code_lines.append("compare.Predicted.head(10)")
            code_lines.append("# Grafica scatter")
            code_lines.append("plt.scatter(y_test,predictions)")
            code_lines.append("# Grafica de distribucion")
            code_lines.append("sn.distplot(y_test - predictions)")
            sc.append(code_lines)

            raise web.seeother('/linearr')
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])
  