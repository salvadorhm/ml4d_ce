import web  # pip install web.py
import webdataminingtool
import csv  # CSV parser

render = web.template.render('application/views/', base="master")

class SvcR:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            return render.svc_r(webdataminingtool.sessions)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  