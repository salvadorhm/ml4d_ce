import web  # pip install web.py
import webdataminingtool
import csv  # CSV parser

render = web.template.render('application/views/', base="master")

class CompareR:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            knn = webdataminingtool.knn
            tree = webdataminingtool.tree
            randomf = webdataminingtool.randomf
            svc = webdataminingtool.svc

            return render.compare_r(knn, tree, randomf, svc)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  