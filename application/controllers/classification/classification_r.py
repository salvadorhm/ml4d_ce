import web  # pip install web.py
import webdataminingtool
import csv  # CSV parser

render = web.template.render('application/views/classification', base="../master")

class ClassificationR:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            title = webdataminingtool.classification['title']
            result = webdataminingtool.classification
            return render.classification_r(title, result)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  