import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np
from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/clean', base="../master")

class Impute:

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self, column):
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
                
            return render.impute(column,nulls,dtypes,unique,mode,mean,median)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

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
            sc.append(code_lines)
            dataframe.to_csv('static/csv/temp.csv', sep=',',index=False)
            raise web.seeother('/detail') 
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  