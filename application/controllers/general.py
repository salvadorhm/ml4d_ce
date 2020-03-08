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

class General:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            values = []
            duplicates = []
            nulls = []
            dtypes = list(dataframe.dtypes)
            for col in cols:
                values.append(dataframe[col].iloc[0:5].tolist())
                duplicates.append(sum(dataframe.duplicated(subset = col)) == 0)
                nulls.append(round(100*(dataframe[col].isnull().sum()/len(dataframe.index)), 2))
            return render.general(cols,values, duplicates,nulls,dtypes)
        except Exception as e:
            print(e.args)

  