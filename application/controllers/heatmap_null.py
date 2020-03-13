import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab

render = web.template.render('application/views/', base="master")

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
            return render.heatmap_null()
        except Exception as e:
            print(e.args)

  