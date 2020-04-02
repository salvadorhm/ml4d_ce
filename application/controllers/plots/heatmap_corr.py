import web  # pip install web.py
import ml4d
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
import seaborn as sn
from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/plots', base="../master")

class HeatmapCorr:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            columns = list(dataframe)
            dtypes = list(dataframe.dtypes)
            cols = []
            for d,c in zip(dtypes,columns):
                if d != "object":
                    cols.append(c)
            return render.heatmap_corr(cols)
        except Exception as e:
            print(e.args)

    def POST(self):
        try:
            images=[]
            dataframe = pd.read_csv(self.file)
            form = web.input(column=[])
            columns = form.column
            dataframe = dataframe[columns]
            correlation = dataframe.corr()
            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            ax = sn.heatmap(correlation,annot=True)
            image_name = "static/images/correlation.png"
            images.append(image_name)
            ax.figure.savefig(image_name)
            fig = ax.get_figure()
            # plt.close('all')

            code_lines = []
            code_lines.append("# Heatmap corr")
            code_lines.append("correlation = dataframe.corr()")
            code_lines.append("sn.heatmap(correlation,annot=True)")
            sc.append(code_lines)

            return render.plots("Heatmap corr", images)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  