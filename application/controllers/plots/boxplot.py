import web  # pip install web.py
import webdataminingtool
import csv  # CSV parser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.mlab as mlab
from matplotlib.pyplot import figure, show

from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/plots', base="../master")

class BoxPlot:

    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            webdataminingtool.sessions = {}
            dataframe = pd.read_csv(self.file)
            columns = list(dataframe)
            return render.boxplot(columns)
        except Exception as e:
            print(e.args)

    def POST(self):
        try:
            dataframe = pd.read_csv(self.file)
            images = []
            form = web.input()
            x_col = form.x
            y_col = form.y
            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            nor = sn.boxplot(x=x_col, y=y_col, data= dataframe)
            image_name = "static/images/boxplot.png"
            images.append(image_name)
            nor.figure.savefig(image_name)
            fig = nor.get_figure()
            #plt.close(fig)

            code_lines = []
            code_lines.append("# Boxplot")
            code_lines.append("sn.boxplot(x='"+ x_col +"', y='"+ y_col + "', data= dataframe)")
            sc.append(code_lines)

            return render.plots("Boxplot",images)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])
 