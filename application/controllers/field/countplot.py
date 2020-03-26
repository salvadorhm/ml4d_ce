import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np
# import statsmodels.api as sm
# import scipy.stats as st
# import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
import seaborn as sn
# from sklearn.metrics import confusion_matrix
# import matplotlib.mlab as mlab

from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/field', base="../master")

class CountPlot:

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
            ax = sn.countplot(data=dataframe, y=column)
            image_name = "static/images/countplot.png"
            print(image_name)
            ax.figure.savefig(image_name)

            code_lines = []
            code_lines.append("# Countplot")
            code_lines.append("sn.countplot(data=dataframe, y='" + column +"')")
            sc.append(code_lines)

            return render.countplot(column)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  