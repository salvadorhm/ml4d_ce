import web  # pip install web.py

urls = (
    '/','application.controllers.index.Index',
    '/columns', 'application.controllers.columns.Columns',
    '/drop/(.+)','application.controllers.drop.Drop',
    '/impute/(.+)','application.controllers.impute.Impute',
    '/replace/(.+)','application.controllers.replace.Replace',
)
app = web.application(urls, globals())

if __name__ == "__main__":
    web.config.debug = False
    app.run()
