import web  # pip install web.py
import webdataminingtool
import csv  # CSV parser

render = web.template.render('application/views/', base="master")

class RandomfR:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            return render.randomf_r(webdataminingtool.sessions)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  