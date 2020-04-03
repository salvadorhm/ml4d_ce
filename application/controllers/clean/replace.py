import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np
from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/clean', base="../master")

class Replace:

    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self, column):
        try:
            dataframe = pd.read_csv(self.file)
            nulls = dataframe[column].isnull().sum()
            dtypes = dataframe[column].dtypes
            unique = dataframe[column].unique().tolist()
            mode = None
            mean = None
            median = None

            '''
                guardar el codigo generado
            '''
            code_lines = []
            code_lines.append("# Describe columna '"+column+"'")
            code_lines.append("dataframe['" + column + "'].describe()" )
            sc.append(code_lines)

            if dtypes == 'object':
                print("Col:{} mode: {}".format(column,dataframe[column].mode()[0]))
                mode = dataframe[column].mode()[0]
                mean = "None"
                median = "None"
            else:
                print("Col:{} mean: {}".format(column,dataframe[column].mean()))
                mode = dataframe[column].mode()[0]
                mean = dataframe[column].mean()
                median = dataframe[column].median()
            return render.replace(column,nulls,dtypes,unique,mode,mean,median)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self, column):
        try:
            form = web.input() # get form data
            column = form['column']
            actual = form['actual']
            new = form['new']
            dataframe = pd.read_csv(self.file)
            dataframe[column].replace({actual : new}, inplace=True, regex=True)
            dataframe.to_csv('static/csv/train.csv', sep=',',index=False)

            code_lines = []
            code_lines.append("# Remplazando '" +  actual + "' por '" + new + "' en la columna '" + column +"'")
            code_lines.append("dataframe['" + column + "'].replace({'" + actual +"':'" + new +"'}, inplace=True, regex=True)")
            sc.append(code_lines)
            raise web.seeother('/detail') 
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  