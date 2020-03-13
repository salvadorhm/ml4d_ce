import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab

render = web.template.render('application/views/', base="master")

class Dummies:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,column):
        try:
            dataframe = pd.read_csv(self.file)
            return render.dummies(column)
        except Exception as e:
            print(e.args)
    
    def POST(self, column):
        try:
            form = web.input() # get form data
            column = form['column']
            dataframe = pd.read_csv(self.file)
            # dummies
            dataframe = pd.get_dummies(dataframe, columns =[column])

            code_lines = []
            code_lines.append("# Getdummies " + column)
            code_lines.append("dataframe = pd.get_dummies(dataframe, columns =['" + column + "'])")

            python_code=open('static/csv/code.py','a+')
            for element in code_lines:
                python_code.write(element+"\n")
            python_code.close()

            # actualiza el archivo csv
            dataframe.to_csv('static/csv/temp.csv', sep=',',index=False)
            print("Dummies")
            raise web.seeother('/detail') 
        except Exception as e:
            print(e.args)
