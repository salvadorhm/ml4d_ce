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

class Drop:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,column):
        try:
            dataframe = pd.read_csv(self.file)
            print(column)
            # dataframe.drop(['Prospect ID'],axis=1,inplace=True)
            # dataframe.drop(['Lead Number'],axis=1,inplace=True)
            # dataframe.to_csv('static/csv/temp.csv', sep=',',index=False)
            return render.drop(column)
        except Exception as e:
            print(e.args)
    
    def POST(self, column):
        try:
            form = web.input() # get form data
            column = form['column']
            dataframe = pd.read_csv(self.file)
            dataframe.drop([column],axis=1,inplace=True)
            dataframe.to_csv('static/csv/temp.csv', sep=',',index=False)
            print("Borrado")
            raise web.seeother('/') 
        except Exception as e:
            print(e.args)

  