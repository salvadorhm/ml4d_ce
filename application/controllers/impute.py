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

            # TODO revisar para guardar las instrucciones
            '''
                guardar el codigo generado
            '''
            code_lines = []
            code_lines.append("# Revisando si tiene NaN la columna '"+column+"'")
            code_lines.append("dataframe['"+column+"'].isnull().sum()")
            code_lines.append("# Describe columna '"+column+"'")
            code_lines.append("dataframe['" + column + "'].describe()" )
            MyFile=open('static/csv/code.py','a+')
            for element in code_lines:
                MyFile.write(element+"\n")
            MyFile.close()

            nulls = dataframe[column].isnull().sum()
            dtypes = dataframe[column].dtypes
            unique = dataframe[column].unique().tolist()

            #TODO revisar conteo de nan
            # unique_counts = dataframe[column].value_counts().tolist()
            # unique_counts = dataframe[column].value_counts(normalize=True).tolist()

            # total_unique = []
            # x=0
            # for i in range(len(unique)):
            #     print(i,x,unique[i],unique_counts[x],len(unique),len(unique_counts))
            #     if str(unique[i]) == 'nan':
            #         print("mmm")
            #     else:
            #         total_unique.append(str(unique[i]) + " : " + str(unique_counts[x]))
            #         x+=1

            mode = None
            mean = None
            median = None
            if dtypes == 'object':
                # print("Col:{} mode: {}".format(column,dataframe[column].mode()[0]))
                mode = dataframe[column].mode()[0]
                mean = "None"
                median = None
            else:
                # print("Col:{} mean: {}".format(column,dataframe[column].mean()))
                mode = dataframe[column].mode()[0]
                mean = dataframe[column].mean()
                median = dataframe[column].median()
                
            return render.impute(column,nulls,dtypes,unique,mode,mean,median)
        except Exception as e:
            print(e.args)

    def POST(self, column):
        try:
            form = web.input() # get form data
            column = form['column']
            nan_value = form['nan_value']
            dataframe = pd.read_csv(self.file)
            dataframe[column].fillna(nan_value, inplace=True)

            
            code_lines = []
            code_lines.append("# Imputando valor a los valores NaN de la columna '"+column+"'")
            code_lines.append("dataframe['"+column+"'].fillna('"+nan_value+"', inplace=True)")
            MyFile=open('static/csv/code.py','a+')
            for element in code_lines:
                MyFile.write(element+"\n")
            MyFile.close()

            dataframe.to_csv('static/csv/temp.csv', sep=',',index=False)
            raise web.seeother('/detail') 
        except Exception as e:
            print(e.args)

  