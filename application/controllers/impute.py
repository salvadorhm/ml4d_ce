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

class Impute:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self, column):
        try:
            dataframe = pd.read_csv(self.file)
            # dataframe.drop(['Prospect ID'],axis=1,inplace=True)
            # dataframe.drop(['Lead Number'],axis=1,inplace=True)
            # dataframe.to_csv('static/csv/temp.csv', sep=',',index=False)
            nulls = dataframe[column].isnull().sum()
            dtypes = dataframe[column].dtypes
            unique = dataframe[column].unique().tolist()
            mode = None
            mean = None

            if dtypes == 'object':
                print("Col:{} mode: {}".format(column,dataframe[column].mode()[0]))
                mode = dataframe[column].mode()[0]
                mean = "None"
            else:
                print("Col:{} mean: {}".format(column,dataframe[column].mean()))
                mode = dataframe[column].mode()[0]
                mean = dataframe[column].mean()
                
            return render.impute(column,nulls,dtypes,unique,mode,mean)
        except Exception as e:
            print(e.args)

    def POST(self, column):
        try:
            form = web.input() # get form data
            column = form['column']
            nan_value = form['nan_value']
            dataframe = pd.read_csv(self.file)
            dataframe[column].fillna(nan_value, inplace=True)
            dataframe.to_csv('static/csv/temp.csv', sep=',',index=False)
            raise web.seeother('/columns') 
        except Exception as e:
            print(e.args)

  