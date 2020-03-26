import web  # pip install web.py
import sys
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

render = web.template.render('application/views/downloads', base="../master")

class Dataset:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            return render.dataset()
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])