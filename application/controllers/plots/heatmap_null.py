import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure, show
import seaborn as sn
from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/plots', base="../master")

class HeatmapNull:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            nor = sn.heatmap(dataframe.isnull())
            image_name = "static/images/heatmap_null.png"
            nor.figure.savefig(image_name)

            code_lines = []
            code_lines.append("# Heatmap nulls")
            code_lines.append("sn.heatmap(dataframe.isnull())")
            sc.append(code_lines)

            return render.plots("Heatmap null", image_name)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  