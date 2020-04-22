import web  # pip install web.py
import ml4d
import pandas as pd
import json
from joblib import load

class API:


    def __init__(self):
        pass

    def GET(self):
        try:
            result = {}
            data = []
            form = web.input()
            m = ml4d.deploy['model']
            cols = ml4d.deploy['cols']
            t = []
            col = {}
            
            # if len(cols) > 1:
            for c in cols:
                t.append(form[c])
                col[c]= form[c]
            data.append(t)
            # else:
            #     c = cols[0]
            #     print(cols,c)
            #     col[c] = form[c]
            #     data.append(form[c])

            xs = pd.DataFrame(data) 
            model = load("static/models/"+m)
            prediction = model.predict(xs)
            
            result['status'] = "200"
            result['message'] = "succeful"
            result['model'] = "static/models/"+m
            result['cols'] = [col]
            result['prediction'] = str(prediction[0])

            return json.dumps(result)
        except Exception as e:
            print(e.args)
            result = {}
            result['status'] = "400"
            result['message'] = e.args
            result['model'] = "static/models/"+m
            result['cols'] = [cols]
            result['prediction'] = None
            return json.dumps(result)
