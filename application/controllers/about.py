import web  # pip install web.py

render = web.template.render('application/views/', base="master")

class About:

    app_version = "0.1.0"  # version de la webapp
    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            return render.about()
        except Exception as e:
            print(e.args)

  