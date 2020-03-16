import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab

from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/', base="master")

class Field:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,field):
        try:
            dataframe = pd.read_csv(self.file)
            
            #describe
            dtypes = dataframe[field].dtypes

            result = dataframe[field].describe()
            describe = result.to_dict()
            describe['nulls'] = dataframe[field].isnull().sum()
            describe['head'] = list(dataframe[field].head())
            describe['unique values'] = dataframe[field].unique().tolist()

            if dtypes != 'object':
                result = dataframe.corr()[field]
                correlation = result.to_dict()
            else:
                correlation = None

            # save code
            code_lines = []
            code_lines.append("# Describe '" + field )
            code_lines.append("dataframe['" + field + "'].describe()" )
            sc.append(code_lines)
            return render.field(describe, correlation, field)
        except Exception as e:
            print(e.args)

  