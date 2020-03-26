import web  # pip install web.py
import os

app_version = "0.7.5"  # version de la webapp

urls = (
    '/','application.controllers.index.Index',
    '/about','application.controllers.about.About',
    '/notebook','application.controllers.notebook.Notebook',
    '/dataset','application.controllers.dataset.Dataset',
    '/general', 'application.controllers.general.General',
    '/head', 'application.controllers.head.Head',

    '/lineary', 'application.controllers.linear.lineary.LinearY',
    '/linearx', 'application.controllers.linear.linearx.LinearX',
    '/linearr', 'application.controllers.linear.linearr.LinearR',

    '/tree_y', 'application.controllers.tree_y.TreeY',
    '/tree_x', 'application.controllers.tree_x.TreeX',
    '/tree_r','application.controllers.tree_r.TreeR',

    '/logisticy', 'application.controllers.logistic.logisticy.LogisticY',
    '/logisticx', 'application.controllers.logistic.logisticx.LogisticX',
    '/logisticr', 'application.controllers.logistic.logisticr.LogisticR',

    '/knn_y', 'application.controllers.knn_y.KnnY',
    '/knn_x', 'application.controllers.knn_x.KnnX',
    '/knn_r','application.controllers.knn_r.KnnR',

    '/svc_y', 'application.controllers.svc_y.SvcY',
    '/svc_x', 'application.controllers.svc_x.SvcX',
    '/svc_r','application.controllers.svc_r.SvcR',

    '/compare_y', 'application.controllers.compare_y.CompareY',
    '/compare_x', 'application.controllers.compare_x.CompareX',
    '/compare_r','application.controllers.compare_r.CompareR',

    '/randomf_y', 'application.controllers.randomf_y.RandomfY',
    '/randomf_x', 'application.controllers.randomf_x.RandomfX',
    '/randomf_r','application.controllers.randomf_r.RandomfR',

    '/boxplot', 'application.controllers.boxplot.BoxPlot',
    '/countplot2', 'application.controllers.countplot2.CountPlot2',
    '/heatmap_null','application.controllers.heatmap_null.HeatmapNull',
    '/pairplot','application.controllers.pairplot.PairPlot',

    '/plots', 'application.controllers.plots.Plots',
    '/detail', 'application.controllers.detail.Detail',
    '/drop/(.+)','application.controllers.drop.Drop',
    '/dummies/(.+)','application.controllers.dummies.Dummies',
    '/field/(.+)','application.controllers.field.Field',
    '/impute/(.+)','application.controllers.impute.Impute',
    '/replace/(.+)','application.controllers.replace.Replace',
    '/countplot/(.+)','application.controllers.countplot.CountPlot',
    '/histogram/(.+)','application.controllers.histogram.Histogram',
    '/describe','application.controllers.describe.Describe',
    '/correlation','application.controllers.correlation.Correlation',
)
webdataminingtool = web.application(urls, globals())
sessions ={"status":"online","url":"localhost:8080"}

def main():
    web.config.debug = False
    print(sessions)
    print(os.path.dirname(os.path.abspath(__file__)))
    webdataminingtool.run()

if __name__ == "__main__":
    
    main()