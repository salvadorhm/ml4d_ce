import web  # pip install web.py

app_version = "0.5.0"  # version de la webapp

urls = (
    '/','application.controllers.index.Index',
    '/about','application.controllers.about.About',
    '/notebook','application.controllers.notebook.Notebook',
    '/dataset','application.controllers.dataset.Dataset',
    '/general', 'application.controllers.general.General',
    '/linealy', 'application.controllers.linealy.LinealY',
    '/linealx/(.+)', 'application.controllers.linealx.LinealX',
    '/detail', 'application.controllers.detail.Detail',
    '/drop/(.+)','application.controllers.drop.Drop',
    '/field/(.+)','application.controllers.field.Field',
    '/impute/(.+)','application.controllers.impute.Impute',
    '/replace/(.+)','application.controllers.replace.Replace',
    '/countplot/(.+)','application.controllers.countplot.CountPlot',
    '/linealr','application.controllers.linealr.LinealR',
    '/histogram/(.+)','application.controllers.histogram.Histogram',
)
app = web.application(urls, globals())

if __name__ == "__main__":
    web.config.debug = False
    app.run()
