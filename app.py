import web  # pip install web.py

app_version = "0.1.0"  # version de la webapp

urls = (
    '/','application.controllers.index.Index',
    '/about','application.controllers.about.About',
    '/general', 'application.controllers.general.General',
    '/detail', 'application.controllers.detail.Detail',
    '/drop/(.+)','application.controllers.drop.Drop',
    '/field/(.+)','application.controllers.field.Field',
    '/impute/(.+)','application.controllers.impute.Impute',
    '/replace/(.+)','application.controllers.replace.Replace',
    '/countplot/(.+)','application.controllers.countplot.CountPlot',
)
app = web.application(urls, globals())

if __name__ == "__main__":
    web.config.debug = False
    app.run()
