import web  # pip install web.py
import ml4d
import csv  # CSV parser

render = web.template.render('application/views/compare', base="../master")

class CompareR:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            knn = ml4d.knn
            tree = ml4d.tree
            randomf = ml4d.randomf
            svc = ml4d.svc

            return render.compare_r(knn, tree, randomf, svc)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  