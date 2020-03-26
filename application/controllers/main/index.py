import web  # pip install web.py
import webdataminingtool  # Application
import shutil # Maneja archivos y directorios.
import cgi,os # Ejecuta un programa en el servidor y despliega su resultado hacia el cliente.
import cgitb # Proporciona un controlador especial para scripts de Python. 
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

from application.controllers.save_code import SaveCode
sc = SaveCode()

cgitb.enable

render = web.template.render('application/views/main', base="../master")


class Index:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,**k):
        try:
            message = None
            app_version = webdataminingtool.app_version
            return render.index(message, app_version)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])


    def POST(self,**k):
        csv_file_import = ""
        form = cgi.FieldStorage
        fileitem = form = web.input(csv_file={})
        filedir = 'static/csv/' # change this to the directory you want to store the file in.
        if form.csv_file.filename == "":
            message= "Not file was selected!!"
            return render.index(message) # render import.html
        else: # to check if the file-object is created
            filepath = form.csv_file.filename.replace('\\','/') # replaces the windows-style slashes with linux ones.
            # TODO cambiar el nombre del archivo por el id de la respuesta a la cual pertenece.
            filename = filepath.split('/')[-1] # splits the and chooses the last part (the filename with extension)
            if os.path.splitext(filepath)[1] == ".csv":  # Extención del archivo
                fn = os.path.basename(form.csv_file.filename)
                webdataminingtool.file = {'filename':fn}
                file = open('static/uploads/'+fn,'wb').write(form.csv_file.file.read(50000000)) # tamaño del archivo
                new_filename = "train.csv"
                shutil.copy('static/uploads/'+fn, filedir + new_filename)


                df = pd.read_csv("static/csv/train.csv")
                train, validation = train_test_split(df, test_size=0.1)
                train.to_csv('static/csv/train.csv', sep=',',index=False)
                validation.to_csv('static/csv/validation.csv', sep=',',index=False)
                

                code_lines = []
                code_lines.append("# Libraries")
                code_lines.append("import csv")
                code_lines.append("import pandas as pd")
                code_lines.append("import numpy as np")
                code_lines.append("import matplotlib.pyplot as plt")
                code_lines.append("import seaborn as sn")
                code_lines.append("import matplotlib.mlab as mlab")
                code_lines.append("from matplotlib.pyplot import figure, show")
                code_lines.append("# Loading Dataframe ")
                code_lines.append("dataframe = pd.read_csv('train')")
                code_lines.append("# Describe dataframe")
                code_lines.append("dataframe.describe()")
                code_lines.append("# Dataframe")
                code_lines.append("dataframe")
                sc.create(code_lines)
                
                raise web.seeother('/general') 
            else: 
                message ="The file it's not a CSV"
                return render.index(message)
