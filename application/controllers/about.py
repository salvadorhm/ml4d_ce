import web  # pip install web.py
import app 

render = web.template.render('application/views/', base="master")

class About:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            app_version = app.app_version
            return render.about(app_version)
        except Exception as e:
            print(e.args)

  