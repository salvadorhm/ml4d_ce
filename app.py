import web  # pip install web.py
import os

app_version = "0.5.9"  # version de la webapp

urls = (
    '/','application.controllers.index.Index',
    '/about','application.controllers.about.About',
    '/notebook','application.controllers.notebook.Notebook',
    '/dataset','application.controllers.dataset.Dataset',
    '/general', 'application.controllers.general.General',
    '/lineary', 'application.controllers.lineary.LinearY',
    '/linearx', 'application.controllers.linearx.LinearX',
    '/detail', 'application.controllers.detail.Detail',
    '/drop/(.+)','application.controllers.drop.Drop',
    '/dummies/(.+)','application.controllers.dummies.Dummies',
    '/field/(.+)','application.controllers.field.Field',
    '/impute/(.+)','application.controllers.impute.Impute',
    '/replace/(.+)','application.controllers.replace.Replace',
    '/countplot/(.+)','application.controllers.countplot.CountPlot',
    '/linearr','application.controllers.linearr.LinearR',
    '/histogram/(.+)','application.controllers.histogram.Histogram',
    '/describe','application.controllers.describe.Describe',
    '/correlation','application.controllers.correlation.Correlation',
)
app = web.application(urls, globals())
sessions ={}
if __name__ == "__main__":
    web.config.debug = False
    
    print(sessions)
    app.run()
