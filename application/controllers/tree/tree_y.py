import web  # pip install web.py
import ml4d
import csv  # CSV parser
import pandas as pd
import numpy as np

render = web.template.render('application/views/tree', base="../master")

class TreeY:

    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            types = list(dataframe.dtypes)
            nulls = list(dataframe.isnull().sum())
            return render.tree_y(cols,types,nulls)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self):
        try:
            form = web.input()
            y = form.column
            ml4d.sessions = {}
            ml4d.sessions['y'] = y
            raise web.seeother('/tree_x')
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])
