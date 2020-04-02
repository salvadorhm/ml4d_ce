import web  # pip install web.py
import ml4d
import csv  # CSV parser
import pandas as pd
import numpy as np

render = web.template.render('application/views/classification', base="../master")

class ClassificationY:

    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self, method):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            types = list(dataframe.dtypes)
            nulls = list(dataframe.isnull().sum())
            if method == "svc":
                title  = "Suport Vector Machine Classifier"
            elif method == "tree":
                title  = "Decision Tree classifier"
            elif method == "randomf":
                title  = "RandomForest Classifier"
            elif method == "knn":
                title  = "KNeighbors classifier"
            return render.classification_y(title,cols,types,nulls)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self, method):
        try:
            form = web.input()
            y = form.column
            ml4d.classification = {}
            if method == "svc":
                ml4d.classification['title'] = "Suport Vector Machine Classifier"
            elif method == "tree":
                ml4d.classification['title'] = "Decision Tree classifier"
            elif method == "randomf":
                ml4d.classification['title'] = "RandomForest Classifier"
            elif method == "knn":
                ml4d.classification['title'] = "KNeighbors classifier"

            ml4d.classification['method'] = method
            ml4d.classification['y'] = y
            raise web.seeother('/classification_x')
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])


  