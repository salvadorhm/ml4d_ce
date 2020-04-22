import web  # pip install web.py
import ml4d
import pandas as pd
from joblib import load

render = web.template.render('application/views/deploy', base="../master")

class Deploy():

    results = []

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            model = ml4d.deploy['model']
            cols = ml4d.deploy['cols']
            prediction = None
            self.results = []
            return render.deploy(cols, prediction)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self):
        try:
            data = []
            form = web.input()
            m = ml4d.deploy['model']
            cols = ml4d.deploy['cols']
            t = []
            input = {}
            for c in cols:
                t.append(form[c])
                input[c] = form[c]
            data.append(t)
            xs = pd.DataFrame(data) 
            model = load("static/models/"+m)
            prediction = model.predict(xs)
            self.results.append([m,prediction[0], input])

            return render.deploy(cols,self.results)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])
