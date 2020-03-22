import web  # pip install web.py
import webdataminingtool
import csv  # CSV parser

render = web.template.render('application/views/', base="master")

class RandomfR:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            result = webdataminingtool.randomf
            result_nor = webdataminingtool.randomf_nor
            result_pca = webdataminingtool.randomf_pca

            return render.randomf_r(result, result_nor, result_pca)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  