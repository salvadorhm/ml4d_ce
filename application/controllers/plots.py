import web  # pip install web.py
import webdataminingtool


render = web.template.render('application/views/', base="master")

class Plots:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self,title,image):
        try:
            return render.plots(title,image)
        except Exception as e:
            print(e.args)

  