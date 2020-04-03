import web  # pip install web.py
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np
import subprocess

from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/clean', base="../master")

class Run:

    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            nulls = list(dataframe.isnull().sum())
            dtypes = list(dataframe.dtypes)

            return render.run(cols,nulls,dtypes)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])
    
    def POST(self):
        try:
            form = web.input() # get form data
            code = form['code']
            mmm = []
            
            mmm.append("import pandas as pd")
            
            mmm.append("df = pd.read_csv('static/csv/train.csv')")
            mmm.append(code)
            mmm.append("df.to_csv('static/csv/train.csv', sep=',',index=False)")
            sc.createCode("run.txt",mmm)
            x = subprocess.call("ipython "+" static/code/run.txt", shell=True)
            message = None
            if x == 0:
                message = "Run code succesfull!!!"
            else:
                message = "Error in code"
            code_lines = []
            code_lines.append("df = pd.read_csv('static/csv/train.csv')")
            code_lines.append(code)
            sc.append(code_lines)
            return render.result(message)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  