import web  # pip install web.py
import webdataminingtool  # Application

render = web.template.render('application/views/main', base="../master")

class Index:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            app_version = webdataminingtool.app_version
            return render.index(app_version)
        except Exception as e:
            print(e.args)
 