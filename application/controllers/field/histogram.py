import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
import seaborn as sn

from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/field', base="../master")

class Histogram:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self, column):
        try:
            dataframe = pd.read_csv(self.file)
            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            nor = sn.distplot(dataframe[column])
            image_name = "static/images/normal.png"
            nor.figure.savefig(image_name)
            # plt.close('all')

            code_lines = []
            code_lines.append("# Histogram de " + column)
            code_lines.append("sn.distplot(dataframe[" + column + "])")
            sc.append(code_lines)

            return render.histogram(column)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  