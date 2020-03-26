import web  # pip install web.py
import sys
import nbformat
from nbformat.v4 import new_notebook, new_code_cell,new_markdown_cell

render = web.template.render('application/views/downloads', base="../master")

class Notebook:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            nb = new_notebook()
            code = []
            with open("static/code/code.py") as f:
                reader = f.readlines()
                block = False
                for row in reader:
                    if row.startswith("#",0):
                        if len(code)>0:
                            nb.cells.append(new_code_cell(code))
                            nb.cells.append(new_markdown_cell(row))
                            code = []
                        else:
                            nb.cells.append(new_markdown_cell(row))
                    elif row.startswith("#",0) == False:
                        code.append(row)
                nb.cells.append(new_code_cell(code))
            nbformat.write(nb, 'static/code/notebook.ipynb')
            return render.notebook(reader)
            # return reader
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])