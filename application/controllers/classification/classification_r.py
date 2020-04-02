import web  # pip install web.py
import ml4d
import csv  # CSV parser

render = web.template.render('application/views/classification', base="../master")

class ClassificationR:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            title = ml4d.classification['title']
            result = ml4d.classification
            return render.classification_r(title, result)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  