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

class BoxPlot:

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            app.sessions = {}
            dataframe = pd.read_csv(self.file)
            columns = list(dataframe)
            return render.boxplot(columns)
        except Exception as e:
            print(e.args)

    def POST(self):
        try:
            dataframe = pd.read_csv(self.file)
            form = web.input()
            x_col = form.x
            y_col = form.y
            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            nor = sn.boxplot(x=x_col, y=y_col, data= dataframe)
            image_name = "static/images/boxplot.png"
            nor.figure.savefig(image_name)
            fig = nor.get_figure()
            #plt.close(fig)
            return render.plots("Boxplot",image_name)
        except Exception as e:
            print(e.args)
 