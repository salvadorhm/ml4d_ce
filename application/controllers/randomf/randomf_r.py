import web  # pip install web.py
import ml4d
import csv  # CSV parser

render = web.template.render('application/views/randomf', base="../master")

class RandomfR:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            result = ml4d.randomf
            result_nor = ml4d.randomf_nor
            result_pca = ml4d.randomf_pca

            return render.randomf_r(result, result_nor, result_pca)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

  