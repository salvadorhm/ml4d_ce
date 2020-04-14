import web  # pip install web.py
import ml4d
import csv  # CSV parser
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import matplotlib.mlab as mlab
from sklearn import tree as tree_pl

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from matplotlib.pyplot import figure, show
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from joblib import dump, load

from application.controllers.save_code import SaveCode
sc = SaveCode()


render = web.template.render('application/views/classification', base="../master")

class ClassificationX():

    file = 'static/csv/train.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            y = ml4d.classification['y']
            method = ml4d.classification['method']
            title = ml4d.classification['title']
            
            columns = []
            types = []
            nulls = []
            correlation = []
            cols.remove(y)

            for row in cols:
                if dataframe[row].dtypes != 'object' and dataframe[y].dtype != 'object':
                    correlation.append(dataframe[y].corr(dataframe[row]))
                    types.append(dataframe[row].dtype)
                    nulls.append(dataframe[row].isnull().sum())
                    columns.append((row))
                else:
                    correlation.append(0)
                    types.append(dataframe[row].dtype)
                    nulls.append(dataframe[row].isnull().sum())
                    columns.append((row))
            return render.classification_x(title,columns,types,nulls,correlation)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self):
        try:
            try:
                filename = ml4d.file['filename']
            except Exception as e:
                filename = "train.csv"
            y = ml4d.classification['y']
            method = ml4d.classification['method']

            form = web.input(column = [''])
            x_cols = form.column
            # ml4d.classification['x']=list(x_cols)
            # ml4d.classification['y']= y
            dataframe = pd.read_csv(self.file)

            df_x = dataframe[x_cols]
            df_y = dataframe[y]

            x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)
            labels = y_train.unique()
            
            if method == "knn":
                '''
                ---------------------------------------------------------------------------
                KNeighbors classifier
                ---------------------------------------------------------------------------
                '''
                # tasa_error = []
                # for i in range(1,100):
                #     model = KNeighborsClassifier(n_neighbors=i, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
                #     model.fit(x_train, y_train)
                #     prediction_i = model.predict(x_test)
                #     tasa_error.append(np.mean(prediction_i != y_test))

                # min = 1
                # n = 0
                # for i in range(len(tasa_error)):
                #     if tasa_error[i] < min:
                #         min = tasa_error[i]
                #         n = i + 1 

                model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)


                title = "KNeighbors Classifier"
                library = "from sklearn.neighbors import KNeighborsClassifier"
                method_model ="model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)"
                # ml4d.classification['n_neighbors']= 5


                # values = range(1,100)

                # figure()
                # width=10
                # height=4
                # figure(figsize=(width,height))
                # plt.plot(values, tasa_error, color="g", marker="o", markerfacecolor="r")
                # plt.xlabel("Optimization")
                # plt.ylabel("Mean error")
                # plt.title("Optimization test")
                # image_name = "static/images/optimization_plot.png"
                # plt.savefig(image_name)
                # ml4d.classification['optimization_plot']= True

            elif method == "tree":
                '''
                ---------------------------------------------------------------------------
                Decision Tree classifier
                ---------------------------------------------------------------------------
                '''
                model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)

                title = "Decision Tree classifier"
                library = "from sklearn.tree import DecisionTreeClassifier"
                method_model ="model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)"

            elif method == "randomf":
                '''
                ---------------------------------------------------------------------------
                RandomForest Classifier
                ---------------------------------------------------------------------------
                '''
                # tasa_error = []
                # for n in range(1,100):
                #     model = RandomForestClassifier(n_estimators=n, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
                #     model.fit(x_train, y_train)
                #     prediction_i = model.predict(x_test)
                #     tasa_error.append(np.mean(prediction_i != y_test))

                # min = 1
                # n = 0
                # for i in range(len(tasa_error)):
                #     if tasa_error[i] < min:
                #         min = tasa_error[i]
                #         n = i + 1 

                # n=80
                model = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=1, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

                title = "RandomForest Classifier"
                library = "from sklearn.ensemble import RandomForestClassifier"
                method_model ="model = RandomForestClassifier(n_estimators=50,criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=1, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)"
                # ml4d.classification['n_estimators']= n

                # values = range(1,100)
                # figure()
                # width=10
                # height=4
                # figure(figsize=(width,height))
                # plt.plot(values, tasa_error, color="g", marker="o", markerfacecolor="r")
                # plt.xlabel("Optimization")
                # plt.ylabel("Mean error")
                # plt.title("Optimization test")
                # image_name = "static/images/optimization_plot.png"
                # plt.savefig(image_name)
                # ml4d.classification['optimization_plot']= True


                
            elif method == "svc":
                '''
                ---------------------------------------------------------------------------
                Suport Vector Machine Classifier
                ---------------------------------------------------------------------------
                '''
                model = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
                title = "Suport Vector Machine Classifier"
                library = "from sklearn.svm import SVC"
                method_model ="model = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)"


            model.fit(x_train,y_train)
            dump(model, "static/models/"+method+".joblib") 
            predictions = model.predict(x_test)
            report = classification_report(y_test, predictions)
            confusion = confusion_matrix(y_test, predictions)
            print("Entrenamiento",confusion)
            data_compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})

            

            code = []
            code.append("import sklearn")
            code.append("\n")
            code.append("import pandas as pd")
            code.append("\n")
            code.append(("from sklearn.metrics import classification_report, confusion_matrix,accuracy_score"))
            code.append("\n")
            code.append("from sklearn.model_selection import train_test_split")
            code.append("\n")
            code.append("from joblib import dump, load")
            code.append("\n")
            code.append(library)
            code.append("\n")
            code.append("dataframe = pd.read_csv('train.csv')")
            code.append("\n")
            code.append("df_x = dataframe["+str(x_cols)+"]")
            code.append("\n")
            code.append("df_y = dataframe['"+y+"']")
            code.append("\n")
            code.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            code.append("\n")
            code.append(method_model)
            code.append("\n")
            code.append("model.fit(x_train,y_train)")
            code.append("\n")
            code.append("# Dump train model to joblib")
            code.append("\n")
            code.append("dump(model," + method + ".joblib)")
            code.append("\n")
            code.append("# Make predictions")
            code.append("\n")
            code.append("predictions = model.predict(x_test)")
            code.append("\n")
            code.append("classification_report(y_test, predictions)")
            code.append("\n")
            code.append("confusion_matrix(y_test, predictions)")

            train_py = []
            train_py.append("__author__ = 'Salvador Hernandez Mendoza'")
            train_py.append("__email__ = 'salvadorhm@gmail.com'")
            train_py.append("__version__ = '"+ml4d.app_version+"'")
            train_py.append("import sklearn")
            train_py.append("import pandas as pd")
            train_py.append(("from sklearn.metrics import classification_report, confusion_matrix,accuracy_score"))
            train_py.append("from sklearn.model_selection import train_test_split")
            train_py.append("from joblib import dump")
            train_py.append(library)
            train_py.append("dataframe = pd.read_csv('train.csv')")
            train_py.append("df_x = dataframe["+str(x_cols)+"]")
            train_py.append("df_y = dataframe['"+y+"']")
            train_py.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            train_py.append(method_model)
            train_py.append("model.fit(x_train,y_train)")
            train_py.append("# Dump train model to joblib")
            train_py.append("dump(model,'" + method + ".joblib')")
            train_py.append("# Make predictions")
            train_py.append("predictions = model.predict(x_test)")
            train_py.append("classification_report(y_test, predictions)")
            train_py.append("confusion_matrix(y_test, predictions)")
            sc.createCode("train.py",train_py)


            test = []
            test.append("# Load model and predict")
            test.append("\n")
            test.append("import sklearn")
            test.append("\n")
            test.append("import pandas as pd")
            test.append("\n")
            test.append("from joblib import load")
            test.append("\n")
            test.append("# Load trained model")
            test.append("\n")
            test.append("model = load('"+method+".joblib')")
            test.append("\n")
            test.append("dataframe = pd.read_csv('validation.csv')")
            test.append("\n")
            test.append("xs = dataframe["+str(x_cols)+"]")
            test.append("predictions = model.predict(xs)")
            test.append("\n")
            test.append("print(predictions)")
           
            predictions_py = []
            predictions_py.append("__author__ = 'Salvador Hernandez Mendoza'")
            predictions_py.append("__email__ = 'salvadorhm@gmail.com'")
            predictions_py.append("__version__ = '"+ml4d.app_version+"'")
            predictions_py.append("import sklearn")
            predictions_py.append("import pandas as pd")
            predictions_py.append("from joblib import load")
            predictions_py.append("model = load('"+method+".joblib')")
            predictions_py.append("dataframe = pd.read_csv('validation.csv')")
            predictions_py.append("xs = dataframe["+str(x_cols)+"]")
            predictions_py.append("predictions = model.predict(xs)")
            predictions_py.append("print(predictions)")
            sc.createCode("predictions.py",predictions_py)

            ml4d.classification = {}
            ml4d.classification['title'] = title
            ml4d.classification['filename']= filename
            ml4d.classification['x']=list(x_cols)
            ml4d.classification['y']= y
            ml4d.classification['Report'] = report
            ml4d.classification['Confusion matrix'] = list(confusion)
            ml4d.classification['Score'] = model.score(x_test,y_test)
            ml4d.classification['Accuracy score'] = accuracy_score(y_test, predictions)
            ml4d.classification['Real test values'] = list(data_compare.Actual.head(10))
            ml4d.classification['Predicted values'] = list(data_compare.Predicted.head(10))
            ml4d.classification['Python train'] = "".join(code)
            ml4d.classification['Python predictions'] = "".join(test)
            ml4d.classification['Model'] = method+".joblib"
            ml4d.classification['train.csv'] = "train.csv"
            # ml4d.classification['13.- validation.csv'] = "validation.csv"
            ml4d.classification['train.py'] = "train.py"
            ml4d.classification['predictions.py'] = "predictions.py"

            figure()
            width=10
            height=4
            figure(figsize=(width,height))

            confusion_columns=[]
            confusion_index=[]
            for variable in labels:
                confusion_columns.append("Predicted:"+str(variable))
                confusion_index.append("Actual:"+str(variable))
            conf_matrix = pd.DataFrame(data=confusion,columns=confusion_columns,index=confusion_index)
            sn.heatmap(conf_matrix, annot=True,  fmt='d',cmap="YlGnBu")
            image_name = "static/images/confusion_plot.png"
            plt.savefig(image_name)
            ml4d.classification['confusion_plot'] = True

          
            if len(labels) <= 2 and dataframe[y].dtypes != 'object':
                probs = model.predict_proba(x_test)  
                probs = probs[:, 1]  
                fper, tper, thresholds = roc_curve(y_test, probs) 
                figure()
                width=10
                height=4
                figure(figsize=(width,height))
                plt.plot(fper, tper, color='orange', label='ROC')
                plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend()
                image_name = "static/images/roc_plot.png"
                plt.savefig(image_name)
                ml4d.classification['roc_plot'] = True
            else:
                ml4d.classification['roc_plot'] = False

            '''
            Update notebook
            '''
            notebook = []
            notebook.append("# " + title)
            notebook.append("import sklearn")
            notebook.append("import pandas as pd")
            notebook.append("from sklearn.metrics import classification_report, confusion_matrix,accuracy_score")
            notebook.append("from sklearn.model_selection import train_test_split")
            notebook.append("from joblib import dump, load")
            notebook.append(library)
            notebook.append("dataframe = pd.read_csv('train.csv')")
            notebook.append("df_x = dataframe["+str(x_cols)+"]")
            notebook.append("df_y = dataframe['"+y+"']")
            notebook.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            notebook.append(method_model)
            notebook.append("model.fit(x_train,y_train)")
            notebook.append("dump(model,'" + method + ".joblib')")
            notebook.append("predictions = model.predict(x_test)")
            notebook.append(("# Classification report"))
            notebook.append("print(classification_report(y_test, predictions))")
            notebook.append(("# Confusion matrix"))
            notebook.append("confusion_matrix(y_test, predictions)")
            notebook.append("# Score")
            notebook.append("model.score(x_test,y_test)")
            notebook.append("# Accuracy score")
            notebook.append("accuracy_score(y_test, predictions)")
            notebook.append("# Data compare")
            notebook.append("data_compare = pd.DataFrame({'Actual':y_test, 'Predicted':predictions})")
            notebook.append("# Compare")
            notebook.append("data_compare")
            notebook.append("# Load fit model and predict")
            notebook.append("import csv")
            notebook.append("import pandas as pd")
            notebook.append("from joblib import dump, load")
            notebook.append("model = load('"+method+".joblib')")
            notebook.append("dataframe_test = pd.read_csv('validation.csv')")
            notebook.append("xs = dataframe_test["+str(x_cols)+"]")
            notebook.append("ys = dataframe_test['"+y+"']")
            notebook.append("predictions = model.predict(xs)")
            notebook.append("data_compare_test = pd.DataFrame({'Actual':ys, 'Predicted':predictions})")
            notebook.append("data_compare_test")
            sc.append(notebook) # actualiza el notebook

            '''
            Usando el modelo
            '''
            # model = load("static/models/"+method+".joblib")
            # dataframe_test = pd.read_csv("static/csv/validation.csv")
            # xs = dataframe_test[x_cols]
            # ys = dataframe_test[y]
            # predictions = model.predict(xs)
            # data_compare_test = pd.DataFrame({"Actual":ys, "Predicted":predictions})
            # print(data_compare_test)

            raise web.seeother('/classification_r')
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

