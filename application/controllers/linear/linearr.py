import web  # pip install web.py
import webdataminingtool
import csv  # CSV parser
# import json  # json parser
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# import scipy.stats as st
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure, show
# import seaborn as sn
# from sklearn.metrics import confusion_matrix
# import matplotlib.mlab as mlab

render = web.template.render('application/views/linear/', base="../master")

class LinearR:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            return render.linearr(webdataminingtool.sessions)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  