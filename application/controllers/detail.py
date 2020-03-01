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

class Detail:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            # dataframe.drop(['Prospect ID'],axis=1,inplace=True)
            # dataframe.drop(['Lead Number'],axis=1,inplace=True)
            # dataframe.to_csv('static/csv/temp.csv', sep=',',index=False)
            head = [dataframe.head()]
            cols = list(dataframe)
            nulls = list(dataframe.isnull().sum())
            dtypes = list(dataframe.dtypes)
            unique = []
            mode = []
            mean = []
            for col in cols:
                unique.append(dataframe[col].unique().tolist())
                # mode.append(dataframe[col].mode())
                # mean.append(dataframe[col].mean())

            for i in range(len(dtypes)):
                if dtypes[i] == 'object':
                    # print("Col:{} mode: {}".format(cols[i],dataframe[cols[i]].mode()[0]))
                    mode.append(dataframe[cols[i]].mode()[0])
                    mean.append("None")
                else:
                    # print("Col:{} mean: {}".format(cols[i],dataframe[cols[i]].mean()))
                    mode.append(dataframe[cols[i]].mode()[0])
                    mean.append(dataframe[cols[i]].mean())
                
            return render.detail(cols,nulls,dtypes,unique,mode,mean)
        except Exception as e:
            print(e.args)

  