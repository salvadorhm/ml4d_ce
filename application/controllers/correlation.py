import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np

from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/', base="master")

class Correlation:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            result = dataframe.corr()
            correlation = result.to_dict()

            code_lines = []
            code_lines.append("# Correlation")
            code_lines.append("dataframe.corr()")
            sc.append(code_lines)

            return render.correlation(correlation)
        except Exception as e:
            print(e.args)

  