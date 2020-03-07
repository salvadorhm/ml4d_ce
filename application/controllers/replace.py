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

class Replace:

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

            '''
                guardar el codigo generado
            '''
            code_lines = []
            code_lines.append("# Describe columna '"+column+"'")
            code_lines.append("dataframe['" + column + "'].describe()" )
            MyFile=open('static/csv/code.py','a+')
            for element in code_lines:
                MyFile.write(element+"\n")
            MyFile.close()

            if dtypes == 'object':
                print("Col:{} mode: {}".format(column,dataframe[column].mode()[0]))
                mode = dataframe[column].mode()[0]
                mean = "None"
            else:
                print("Col:{} mean: {}".format(column,dataframe[column].mean()))
                mode = dataframe[column].mode()[0]
                mean = dataframe[column].mean()
                
            return render.replace(column,nulls,dtypes,unique,mode,mean)
        except Exception as e:
            print(e.args)

    def POST(self, column):
        try:
            form = web.input() # get form data
            column = form['column']
            actual = form['actual']
            new = form['new']
            dataframe = pd.read_csv(self.file)
            dataframe[column].replace({actual : new}, inplace=True, regex=True)
            dataframe.to_csv('static/csv/temp.csv', sep=',',index=False)

            '''
                guardar el codigo generado
            '''
            code_lines = []
            code_lines.append("# Remplazando '" +  actual + "' por '" + new + "' en la columna '" + column +"'")
            code_lines.append("dataframe['" + column + "'].replace({'" + actual +"':'" + new +"'}, inplace=True, regex=True)")
            MyFile=open('static/csv/code.py','a+')
            for element in code_lines:
                MyFile.write(element+"\n")
            MyFile.close()

            raise web.seeother('/detail') 
        except Exception as e:
            print(e.args)

  