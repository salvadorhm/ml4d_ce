import web  # pip install web.py
import app  # Application
import shutil # Maneja archivos y directorios.
import cgi,os # Ejecuta un programa en el servidor y despliega su resultado hacia el cliente.
import cgitb # Proporciona un controlador especial para scripts de Python. 
import os, sys

cgitb.enable

render = web.template.render('application/views/', base="master")

class Index:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,**k):
        try:
            message = None
            app_version = app.app_version
            return render.index(message, app_version)
        except Exception as e:
            print(e.args)


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
                file = open('static/csv/'+fn,'wb').write(form.csv_file.file.read(50000000)) # tamaño del archivo
                new_filename = "temp.csv"
                shutil.copy('static/csv/'+fn, filedir + new_filename)
                code_lines = []
                code_lines.append("# Librerias")
                code_lines.append("import pandas as pd")
                code_lines.append("import numpy as np")
                code_lines.append("import statsmodels.api as sm")
                code_lines.append("import scipy.stats as st")
                code_lines.append("import matplotlib.pyplot as plt")
                code_lines.append("import seaborn as sn")
                code_lines.append("from sklearn.metrics import confusion_matrix")
                code_lines.append("import matplotlib.mlab as mlab")
                code_lines.append("# Creando el Dataframe para trabajar")
                code_lines.append("dataframe = pd.read_csv('"+filename+"')")
                MyFile=open('static/csv/code.py','a+')
                for element in code_lines:
                    MyFile.write(element+"\n")
                MyFile.close()
                raise web.seeother('/head') 
            else: 
                message ="The file it's not a CSV"
                return render.index(message)
