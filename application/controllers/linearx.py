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
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import figure, show
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


render = web.template.render('application/views/', base="master")

class LinearX:

    file = 'static/csv/temp.csv'  # define el archivo donde se amodelacenan los datos

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
            # print(index)
            # print(len(cols))
            # print(len(types))
            # print(len(nulls))
            # print(len(correlation))
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

    def POST(self):
        # try:
            y = app.sessions['y']
            form = web.input(column = [''])
            # columns = form.column
            x_cols = form.column
            app.sessions['x']=list(x_cols)
            dataframe = pd.read_csv(self.file)
            # columns.append(y)
            # df = dataframe[columns]

            df_x = dataframe[x_cols]
            df_y = dataframe[y]

            df_nor_x = (df_x - df_x.mean())/df_x.std()
            df_nor_y = (df_y - df_y.mean())/df_y.std()
            

            x_train, x_test, y_train, y_test = train_test_split(df_nor_x,df_nor_y,test_size=0.3,random_state=42)
            
            # x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)

            model = LinearRegression()
            model.fit(x_train,y_train)

            predictions = model.predict(x_test)

            if len(list(x_test)) == 1:
                print("graficar")
                figure()
                width=20
                height=8
                figure(figsize=(width,height))
                # TODO grafica lineal simple, una sola x
                ax = plt.scatter(y_test,predictions)
                plt.plot(x_test,predictions,"r")
                image_name = "static/images/lineal.png"
                ax.figure.savefig(image_name)
                fig = ax.get_figure()
                #plt.close(fig)
            else:
                figure()
                width=20
                height=8
                figure(figsize=(width,height))
                # TODO grafica lineal simple, una sola x
                ax = plt.scatter(y_test,predictions)
                # plt.plot(x_test,predictions,"r")
                image_name = "static/images/lineal.png"
                ax.figure.savefig(image_name)
                fig = ax.get_figure()
                #plt.close(fig)


            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            nor = sn.distplot(y_test - predictions)
            image_name = "static/images/histogram.png"
            nor.figure.savefig(image_name)
            fig = nor.get_figure()
            #plt.close(fig)


            app.sessions['Coefficients'] = str(model.coef_)
            app.sessions['Independent term'] = model.intercept_
            app.sessions['Mean squared error'] = mean_squared_error(y_test, predictions)
            app.sessions['Mean absolute error'] = mean_absolute_error(y_test, predictions)
            app.sessions['Variance'] = r2_score(y_test, predictions)

            compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})
            app.sessions['Actual test values'] = list(compare.Actual.head())
            app.sessions['Predicted values'] = list(compare.Predicted.head())

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


            python_code=open('static/csv/code.py','a+')
            for element in code_lines:
                python_code.write(element+"\n")
            python_code.close()

            raise web.seeother('/linearr')
        # except Exception as e:
            # print(e.args)
  