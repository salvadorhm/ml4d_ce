import web  # pip install web.py

urls = (
    '/','application.controllers.index.Index',
    '/columns', 'application.controllers.columns.Columns',
    '/drop/(.+)','application.controllers.drop.Drop'
)
app = web.application(urls, globals())

if __name__ == "__main__":
    web.config.debug = False
    app.run()
