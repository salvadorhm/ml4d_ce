import web  # pip install web.py
import ml4d
import csv  # CSV parser

render = web.template.render('application/views/logistic', base="../master")

class LogisticR:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            return render.logisticr(ml4d.sessions)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  