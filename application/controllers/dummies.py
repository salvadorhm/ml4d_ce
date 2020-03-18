import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np

from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/', base="master")

class Dummies:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,column):
        try:
            dataframe = pd.read_csv(self.file)
            return render.dummies(column)
        except Exception as e:
            print(e.args)
    
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
            dataframe.to_csv('static/csv/temp.csv', sep=',',index=False)
            raise web.seeother('/detail') 
        except Exception as e:
            print(e.args)
