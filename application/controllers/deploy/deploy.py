import web  # pip install web.py
import ml4d
import pandas as pd
from joblib import load
from application.controllers.save_code import SaveCode
sc = SaveCode()

render = web.template.render('application/views/deploy', base="../master")

class Deploy():

    results = []

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:

            prediction = None
            cols = sc.readCols()
            model = sc.readModel()
            if cols == None or model == None:
                return render.error("Not model trained, first train a model")
            else:
                return render.deploy(cols, prediction)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self):
        try:
            data = []
            form = web.input()
            m = sc.readModel()
            cols = sc.readCols()
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
