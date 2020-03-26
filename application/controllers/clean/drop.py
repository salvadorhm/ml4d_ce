import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np

from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/clean', base="../master")

class Drop:

    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,column):
        try:
            dataframe = pd.read_csv(self.file)
            return render.drop(column)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])
    
    def POST(self, column):
        try:
            form = web.input() # get form data
            column = form['column']
            dataframe = pd.read_csv(self.file)
            dataframe.drop([column],axis=1,inplace=True)

            dataframe.to_csv('static/csv/train.csv', sep=',',index=False)

            code_lines = []
            code_lines.append("# Drop")
            code_lines.append("dataframe.drop(['"+ column + "'],axis=1,inplace=True)")
            sc.append(code_lines)
            raise web.seeother('/detail') 
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  