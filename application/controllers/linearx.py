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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


render = web.template.render('application/views/', base="master")

class LinearX:

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,y):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
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

    def POST(self,y):
        try:
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
            

            # x_train, x_test, y_train, y_test = train_test_split(df_nor_x,df_nor_y,test_size=0.4,random_state=101)
            x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.4,random_state=101)

            # print(list(df_x))
            # print(list(df_y))
            # # df_nom = (df - df.mean())/df.std()

            # df_nom = (df - df.min())/(df.max() - df.min())

            # x = df_nom[x_cols]
            # y = df_nom[y]
            lm = LinearRegression()
            lm.fit(x_train,y_train)
            # lm.fit(df_x,df_y)
            predictions = lm.predict(x_train)
            # predictions = lm.predict(df_x)
            if len(list(x_train)) == 1:
                print("graficar")
                figure()
                width=20
                height=8
                figure(figsize=(width,height))
                # TODO grafica lineal simple, una sola x
                ax = plt.scatter(x_train,y_train)
                # ax = plt.scatter(y_train,predictions)
                # ax = plt.scatter(df_y,predictions)
                plt.plot(x_train,predictions,"r")
                # # plt.scatter(y,predictions)
                image_name = "static/images/lineal.png"
                # print(image_name)
                ax.figure.savefig(image_name)

            app.sessions['Coefficients'] = str(lm.coef_)
            app.sessions['Independent term'] = lm.intercept_
            app.sessions['Mean squared error'] = mean_squared_error(y_train, predictions)
            # app.sessions['Mean squared error'] = mean_squared_error(df_y, predictions)
            app.sessions['Variance'] = r2_score(y_train, predictions)
            # app.sessions['Variance'] = r2_score(df_y, predictions)
            # app.sessions['Predictions'] = str(predictions)

            mmmm = pd.DataFrame({"Actual":y_train, "Predicted":predictions})
            app.sessions['Actual values'] = list(mmmm.Actual.head())
            app.sessions['Predicted values'] = list(mmmm.Predicted.head())

            # # Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
            # print('Coefficients: \n', lm.coef_)
            # # Este es el valor donde corta el eje Y (en X=0)
            # print('Independent term: \n', lm.intercept_)
            # # Error Cuadrado Medio
            # print("Mean squared error: %.2f" % mean_squared_error(y, predictions))
            # # Puntaje de Varianza. El mejor puntaje es un 1.0
            # print('Variance score: %.2f' % r2_score(y, predictions))

            raise web.seeother('/linearr')
        except Exception as e:
            print(e.args)


  