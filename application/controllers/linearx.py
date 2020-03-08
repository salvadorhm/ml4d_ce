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


render = web.template.render('application/views/', base="master")

class LinearX:

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,y):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            types = list(dataframe.dtypes)
            nulls = list(dataframe.isnull().sum())
            cols.remove(y)
            return render.linearx(cols,types,nulls)
        except Exception as e:
            print(e.args)

    def POST(self,y):
        try:
            form = web.input(column = [''])
            columns = form.column
            x_cols = form.column
            app.sessions['x']=str(x_cols)
            dataframe = pd.read_csv(self.file)
            columns.append(y)
            df = dataframe[columns]
            # df_nom = (df - df.mean())/df.std()

            df_nom = (df - df.min())/(df.max() - df.min())

            x = df_nom[x_cols]
            y = df_nom[y]
            lm = LinearRegression()
            lm.fit(x,y)
            predictions = lm.predict(x)
            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            ax = plt.scatter(y,predictions)
            # plt.scatter(y,predictions)
            image_name = "static/images/lineal.png"
            print(image_name)
            ax.figure.savefig(image_name)

            app.sessions['coefficients']=str(lm.coef_)
            app.sessions['independent_term']=lm.intercept_
            app.sessions['mean_squared_error']=mean_squared_error(y, predictions)
            app.sessions['variance']=r2_score(y, predictions)

            # Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
            print('Coefficients: \n', lm.coef_)
            # Este es el valor donde corta el eje Y (en X=0)
            print('Independent term: \n', lm.intercept_)
            # Error Cuadrado Medio
            print("Mean squared error: %.2f" % mean_squared_error(y, predictions))
            # Puntaje de Varianza. El mejor puntaje es un 1.0
            print('Variance score: %.2f' % r2_score(y, predictions))

            raise web.seeother('/linearr')
        except Exception as e:
            print(e.args)


  