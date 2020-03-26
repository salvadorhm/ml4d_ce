import web  # pip install web.py
import webdataminingtool
import csv  # CSV parser
import pandas as pd
import numpy as np

render = web.template.render('application/views/classification', base="../master")

class ClassificationY:

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self, method):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            types = list(dataframe.dtypes)
            nulls = list(dataframe.isnull().sum())
            return render.classification_y(method,cols,types,nulls)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self, method):
        try:
            form = web.input()
            y = form.column
            webdataminingtool.classification = {}
            webdataminingtool.classification['method'] = method
            webdataminingtool.classification['y'] = y
            raise web.seeother('/classification_x')
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])


  