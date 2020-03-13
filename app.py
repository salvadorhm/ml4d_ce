import web  # pip install web.py
import os

app_version = "0.5.12"  # version de la webapp

urls = (
    '/','application.controllers.index.Index',
    '/about','application.controllers.about.About',
    '/notebook','application.controllers.notebook.Notebook',
    '/dataset','application.controllers.dataset.Dataset',
    '/general', 'application.controllers.general.General',
    '/head', 'application.controllers.head.Head',

    '/lineary', 'application.controllers.lineary.LinearY',
    '/linearx', 'application.controllers.linearx.LinearX',
    '/linearr','application.controllers.linearr.LinearR',

    '/logisticy', 'application.controllers.logisticy.LogisticY',
    '/logisticx', 'application.controllers.logisticx.LogisticX',
    '/logisticr','application.controllers.logisticr.LogisticR',

    '/boxplot', 'application.controllers.boxplot.BoxPlot',
    '/countplot2', 'application.controllers.countplot2.CountPlot2',
    '/plots', 'application.controllers.plots.Plots',
    '/detail', 'application.controllers.detail.Detail',
    '/drop/(.+)','application.controllers.drop.Drop',
    '/dummies/(.+)','application.controllers.dummies.Dummies',
    '/field/(.+)','application.controllers.field.Field',
    '/impute/(.+)','application.controllers.impute.Impute',
    '/replace/(.+)','application.controllers.replace.Replace',
    '/countplot/(.+)','application.controllers.countplot.CountPlot',
    '/histogram/(.+)','application.controllers.histogram.Histogram',
    '/heatmap_null','application.controllers.heatmap_null.HeatmapNull',
    '/describe','application.controllers.describe.Describe',
    '/correlation','application.controllers.correlation.Correlation',
)
app = web.application(urls, globals())
sessions ={}
if __name__ == "__main__":
    web.config.debug = False
    
    print(sessions)
    app.run()
