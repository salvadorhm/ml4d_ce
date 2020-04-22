import web  # pip install web.py
import ml4d
import csv  # CSV parser
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import matplotlib.mlab as mlab
from sklearn import tree as tree_pl

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from matplotlib.pyplot import figure, show
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from joblib import dump, load

from application.controllers.save_code import SaveCode
sc = SaveCode()


render = web.template.render('application/views/deploy', base="../master")

class Deploy():

    results = []

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            model = ml4d.deploy['model']
            cols = ml4d.deploy['cols']
            prediction = None
            self.results = []
            return render.deploy(cols, prediction)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self):
        try:
            data = []
            form = web.input()
            m = ml4d.deploy['model']
            cols = ml4d.deploy['cols']
            t = []
            for c in cols:
                t.append(form[c])
            data.append(t)
            xs = pd.DataFrame(data) 
            print(xs)
            model = load("static/models/"+m)
            prediction = model.predict(xs)
            self.results.append([prediction[0], data])

            return render.deploy(cols,self.results)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

