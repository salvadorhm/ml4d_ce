import web  # pip install web.py
import webdataminingtool
import csv  # CSV parser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.pyplot import figure, show

from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/plots', base="../master")

class CountPlot2:

    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            webdataminingtool.sessions = {}
            dataframe = pd.read_csv(self.file)
            columns = list(dataframe)
            return render.countplot2(columns)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self):
        try:
            dataframe = pd.read_csv(self.file)
            # form = web.input(column = [''])
            form = web.input()
            x_col = form.x
            hue_col = form.hue
            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            nor = sn.countplot(x=x_col, hue=hue_col, data= dataframe)
            image_name = "static/images/countplot2.png"
            nor.figure.savefig(image_name)
            fig = nor.get_figure()
            #plt.close(fig)

            code_lines = []
            code_lines.append("# Countplot")
            code_lines.append("sn.countplot(x='"+ x_col +"', hue='"+ hue_col + "', data= dataframe)")
            sc.append(code_lines)

            return render.plots("Countplot",image_name)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])
 