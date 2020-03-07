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

render = web.template.render('application/views/', base="master")

class LinealY:

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            return render.linealy(cols)
        except Exception as e:
            print(e.args)

    def POST(self):
        try:
            form = web.input()
            y = form.column
            print(y)
            # dataframe = pd.read_csv(self.file)
            # cols = list(dataframe)
            # return render.lineal(cols)
            raise web.seeother('/linealx/'+y)
        except Exception as e:
            print(e.args)


  