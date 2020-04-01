import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np

from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/clean', base="../master")

class Dummies:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,column):
        try:
            dataframe = pd.read_csv(self.file)
            code_lines = []
            code_lines.append("# Revisando si tiene NaN la columna '"+column+"'")
            code_lines.append("dataframe['"+column+"'].isnull().sum()")
            code_lines.append("# Describe columna '"+column+"'")
            code_lines.append("dataframe['" + column + "'].describe()" )
            sc.append(code_lines)

            nulls = dataframe[column].isnull().sum()
            dtypes = dataframe[column].dtypes
            unique = dataframe[column].unique().tolist()

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
            return render.dummies(column,nulls,dtypes,unique,mode,mean,median)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])
    
    def POST(self, column):
        try:
            form = web.input() # get form data
            column = form['column']
            dataframe = pd.read_csv(self.file)
            # dummies
            dataframe = pd.get_dummies(dataframe, columns =[column])

            code_lines = []
            code_lines.append("# Getdummies " + column)
            code_lines.append("dataframe = pd.get_dummies(dataframe, columns =['" + column + "'])")
            sc.append(code_lines)

            # actualiza el archivo csv
            dataframe.to_csv('static/csv/train.csv', sep=',',index=False)
            raise web.seeother('/detail') 
        except Exception as e:
            print(e.args)
