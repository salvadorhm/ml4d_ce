import web  # pip install web.py
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


render = web.template.render('application/views/', base="master")

class LinealX:

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,y):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            return render.linealx(cols)
        except Exception as e:
            print(e.args)

    def POST(self,y_col):
        try:
            form = web.input(column = [''])
            x_cols = form.column
            print(x_cols)
            dataframe = pd.read_csv(self.file)
            x = dataframe[x_cols]
            y = dataframe[y_col]
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
            
            # dataframe = pd.read_csv(self.file)
            # cols = list(dataframe)
            # return render.lineal(cols)
            raise web.seeother('/linealr')
        except Exception as e:
            print(e.args)


  