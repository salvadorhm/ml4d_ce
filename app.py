import web  # pip install web.py

app_version = "0.5.0"  # version de la webapp

urls = (
    '/','application.controllers.index.Index',
    '/about','application.controllers.about.About',
    '/notebook','application.controllers.notebook.Notebook',
    '/dataset','application.controllers.dataset.Dataset',
    '/general', 'application.controllers.general.General',
    '/lineary', 'application.controllers.lineary.LinearY',
    '/linearx/(.+)', 'application.controllers.linearx.LinearX',
    '/detail', 'application.controllers.detail.Detail',
    '/drop/(.+)','application.controllers.drop.Drop',
    '/field/(.+)','application.controllers.field.Field',
    '/impute/(.+)','application.controllers.impute.Impute',
    '/replace/(.+)','application.controllers.replace.Replace',
    '/countplot/(.+)','application.controllers.countplot.CountPlot',
    '/linearr','application.controllers.linearr.LinearR',
    '/histogram/(.+)','application.controllers.histogram.Histogram',
    '/correlation','application.controllers.correlation.Correlation',
)
app = web.application(urls, globals())

sessions ={"user":'guess'}

if __name__ == "__main__":
    web.config.debug = False
    print(sessions)
    app.run()
