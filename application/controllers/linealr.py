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

class LinealR:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            # dataframe = pd.read_csv(self.file)
            # figure()
            # width=20
            # height=8
            # figure(figsize=(width,height))
            # ax = sn.countplot(data=dataframe, y=column)
            # # plt.xticks(rotation=70)
            # # plt.rcParams['xtick.labelsize'] = 15
            # # plt.rcParams['axes.labelsize'] = 200
            # # ax = sn.countplot(dataframe[column])
            # image_name = "static/images/countplot.png"
            # print(image_name)
            # ax.figure.savefig(image_name)
            return render.linealr()
        except Exception as e:
            print(e.args)

  