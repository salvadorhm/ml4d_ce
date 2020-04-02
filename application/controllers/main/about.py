import web  # pip install web.py
import ml4d 

render = web.template.render('application/views/main', base="../master")

class About:

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            app_version = ml4d.app_version
            return render.about(app_version)
        except Exception as e:
            print(e.args)

  