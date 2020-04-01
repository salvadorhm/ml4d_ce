import web  # pip install web.py
import os

app_version = "0.70"  # version de la webapp

urls = (
    '/','application.controllers.main.index.Index',
    '/upload','application.controllers.main.upload.Upload',
    '/about','application.controllers.main.about.About',

    '/notebook','application.controllers.downloads.notebook.Notebook',
    '/dataset','application.controllers.downloads.dataset.Dataset',

    '/general', 'application.controllers.statistics.general.General',
    '/describe','application.controllers.statistics.describe.Describe',
    '/correlation','application.controllers.statistics.correlation.Correlation',

    '/classification_y/(.+)', 'application.controllers.classification.classification_y.ClassificationY',
    '/classification_x', 'application.controllers.classification.classification_x.ClassificationX',
    '/classification_r', 'application.controllers.classification.classification_r.ClassificationR',

    '/lineary', 'application.controllers.linear.lineary.LinearY',
    '/linearx', 'application.controllers.linear.linearx.LinearX',
    '/linearr', 'application.controllers.linear.linearr.LinearR',

    '/tree_y', 'application.controllers.tree.tree_y.TreeY',
    '/tree_x', 'application.controllers.tree.tree_x.TreeX',
    '/tree_r', 'application.controllers.tree.tree_r.TreeR',

    '/logisticy', 'application.controllers.logistic.logisticy.LogisticY',
    '/logisticx', 'application.controllers.logistic.logisticx.LogisticX',
    '/logisticr', 'application.controllers.logistic.logisticr.LogisticR',

    '/knn_y', 'application.controllers.knn.knn_y.KnnY',
    '/knn_x', 'application.controllers.knn.knn_x.KnnX',
    '/knn_r', 'application.controllers.knn.knn_r.KnnR',

    '/svc_y', 'application.controllers.svc.svc_y.SvcY',
    '/svc_x', 'application.controllers.svc.svc_x.SvcX',
    '/svc_r','application.controllers.svc.svc_r.SvcR',

    '/compare_y', 'application.controllers.compare.compare_y.CompareY',
    '/compare_x', 'application.controllers.compare.compare_x.CompareX',
    '/compare_r', 'application.controllers.compare.compare_r.CompareR',

    '/randomf_y', 'application.controllers.randomf.randomf_y.RandomfY',
    '/randomf_x', 'application.controllers.randomf.randomf_x.RandomfX',
    '/randomf_r','application.controllers.randomf.randomf_r.RandomfR',

    '/boxplot', 'application.controllers.plots.boxplot.BoxPlot',
    '/countplot2', 'application.controllers.plots.countplot2.CountPlot2',
    '/heatmap_null','application.controllers.plots.heatmap_null.HeatmapNull',
    '/heatmap_corr','application.controllers.plots.heatmap_corr.HeatmapCorr',
    '/histogram_plot','application.controllers.plots.histogram_plot.HistogramPlot',
    '/plots', 'application.controllers.plots.plots.Plots',

    '/detail', 'application.controllers.clean.detail.Detail',
    '/drop/(.+)','application.controllers.clean.drop.Drop',
    '/dummies/(.+)','application.controllers.clean.dummies.Dummies',
    '/rename/(.+)','application.controllers.clean.rename.Rename',
    '/impute/(.+)','application.controllers.clean.impute.Impute',
    '/replace/(.+)','application.controllers.clean.replace.Replace',

    '/field/(.+)','application.controllers.field.field.Field',
    '/countplot/(.+)','application.controllers.field.countplot.CountPlot',
    '/histogram/(.+)','application.controllers.field.histogram.Histogram',

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