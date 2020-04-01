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

class HistogramPlot:

    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            webdataminingtool.sessions = {}
            dataframe = pd.read_csv(self.file)
            columns = list(dataframe)
            dtypes = list(dataframe.dtypes)
            cols = []
            for d,c in zip(dtypes,columns):
                if d != "object":
                    cols.append(c)
            return render.histogram_plot(cols)
        except Exception as e:
            print(e.args)

    def POST(self):
        try:
            dataframe = pd.read_csv(self.file)
            form = web.input(column=[])
            columns = form.column
            print("columns",columns)
            images = []
            i = 0 # avoid bad file names
            for column in columns:
                try:
                    figure()
                    width=10
                    height=4
                    figure(figsize=(width,height))
                    nor = sn.distplot(dataframe[column])
                    image_name = "static/images/histogram"+str(i)+".png"
                    i += 1
                    images.append(image_name)
                    nor.figure.savefig(image_name)
                    plt.close('all')
                except Exception as e:
                    print(e.args[0])

            return render.plots("Histograms",images)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])
 