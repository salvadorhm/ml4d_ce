import web  # pip install web.py
import webdataminingtool
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

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            y = webdataminingtool.classification['y']
            method = webdataminingtool.classification['method']
            title = webdataminingtool.classification['title']
            
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
        # try:
            try:
                filename = webdataminingtool.file['filename']
            except Exception as e:
                filename = "temp.csv"
            y = webdataminingtool.classification['y']
            method = webdataminingtool.classification['method']

            form = web.input(column = [''])
            x_cols = form.column
            webdataminingtool.classification['x']=list(x_cols)
            webdataminingtool.classification['y']= y
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
                tasa_error = []
                for i in range(1,100):
                    model = KNeighborsClassifier(n_neighbors=i, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
                    model.fit(x_train, y_train)
                    prediction_i = model.predict(x_test)
                    tasa_error.append(np.mean(prediction_i != y_test))

                min = 1
                n = 0
                for i in range(len(tasa_error)):
                    if tasa_error[i] < min:
                        min = tasa_error[i]
                        n = i + 1 

                model = KNeighborsClassifier(n_neighbors=n, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)


                title = "KNeighbors Classifier"
                library = "from sklearn.neighbors import KNeighborsClassifier"
                method_model ="model = KNeighborsClassifier(n_neighbors="+str(n)+", weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)"
                webdataminingtool.classification['n_neighbors']= n


                values = range(1,100)

                figure()
                width=10
                height=4
                figure(figsize=(width,height))
                plt.plot(values, tasa_error, color="g", marker="o", markerfacecolor="r")
                plt.xlabel("Optimization")
                plt.ylabel("Mean error")
                plt.title("Optimization test")
                image_name = "static/images/optimization_plot.png"
                plt.savefig(image_name)
                webdataminingtool.classification['optimization_plot']= True

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
                tasa_error = []
                for n in range(1,100):
                    model = RandomForestClassifier(n_estimators=n, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
                    model.fit(x_train, y_train)
                    prediction_i = model.predict(x_test)
                    tasa_error.append(np.mean(prediction_i != y_test))

                min = 1
                n = 0
                for i in range(len(tasa_error)):
                    if tasa_error[i] < min:
                        min = tasa_error[i]
                        n = i + 1 

                # n=80
                model = RandomForestClassifier(n_estimators=n, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=1, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

                title = "RandomForest Classifier"
                library = "from sklearn.ensemble import RandomForestClassifier"
                method_model ="model = RandomForestClassifier(n_estimators=" +str(n)+",criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=1, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)"
                webdataminingtool.classification['n_estimators']= n

                values = range(1,100)
                figure()
                width=10
                height=4
                figure(figsize=(width,height))
                plt.plot(values, tasa_error, color="g", marker="o", markerfacecolor="r")
                plt.xlabel("Optimization")
                plt.ylabel("Mean error")
                plt.title("Optimization test")
                image_name = "static/images/optimization_plot.png"
                plt.savefig(image_name)
                webdataminingtool.classification['optimization_plot']= True


                
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
            code.append("import csv")
            code.append("\n")
            code.append("import pandas as pd")
            code.append("\n")
            code.append(("from sklearn.metrics import classification_report, confusion_matrix,accuracy_score"))
            code.append("\n")
            code.append("from sklearn.model_selection import train_test_split")
            code.append("\n")
            code.append(library)
            code.append("\n")
            code.append("dataframe = pd.read_csv("+filename+")")
            code.append("\n")
            code.append("df_x = dataframe["+str(x_cols)+"]")
            code.append("\n")
            code.append("df_y = dataframe['"+y+"']")
            code.append("\n")
            code.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            code.append("\n")
            code.append("from joblib import dump, load")
            code.append("\n")
            code.append(method_model)
            code.append("\n")
            code.append("model.fit(x_train,y_train)")
            code.append("\n")
            code.append("dump(model," + method + ".joblib)")
            code.append("\n")
            code.append("predictions = model.predict(x_test)")
            code.append("\n")
            code.append("classification_report(y_test, predictions)")
            code.append("\n")
            code.append("confusion_matrix(y_test, predictions)")
           
            test = []
            test.append("# Load fit model and predict")
            test.append("\n")
            test.append("import csv")
            test.append("\n")
            test.append("import pandas as pd")
            test.append("\n")
            test.append(("from sklearn.metrics import classification_report, confusion_matrix,accuracy_score"))
            test.append("\n")
            test.append("model = load('"+method+".joblib')")
            test.append("\n")
            test.append("dataframe_test = pd.read_csv('test.csv')")
            test.append("\n")
            test.append("xs = dataframe["+str(x_cols)+"]")
            test.append("\n")
            test.append("ys = dataframe['"+y+"']")
            test.append("\n")
            test.append("predictions = model.predict(xs)")
            test.append("\n")
            test.append("data_compare_test = pd.DataFrame({'Actual':ys, 'Predicted':predictions})")
            test.append("\n")
            test.append("data_compare_test")

            webdataminingtool.classification['filename']= filename
            webdataminingtool.classification['x']=list(x_cols)
            webdataminingtool.classification['y']= y
            webdataminingtool.classification['Report'] = report
            webdataminingtool.classification['Confusion matrix'] = list(confusion)
            webdataminingtool.classification['Score'] = model.score(x_test,y_test)
            webdataminingtool.classification['Accuracy score'] = accuracy_score(y_test, predictions)
            webdataminingtool.classification['Real test values'] = list(data_compare.Actual.head(10))
            webdataminingtool.classification['Predicted values'] = list(data_compare.Predicted.head(10))
            webdataminingtool.classification['Python'] = "".join(code)
            webdataminingtool.classification['Model'] = method+".joblib"
            webdataminingtool.classification['Python_test'] = "".join(test)

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
            webdataminingtool.classification['confusion_plot'] = True

          
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
                webdataminingtool.classification['roc_plot'] = True
            else:
                webdataminingtool.classification['roc_plot'] = False

            '''
            Update notebook
            '''
            notebook = []
            notebook.append("# " + title)
            notebook.append("import csv")
            notebook.append("import pandas as pd")
            notebook.append("from sklearn.metrics import classification_report, confusion_matrix,accuracy_score")
            notebook.append("from sklearn.model_selection import train_test_split")
            notebook.append("from joblib import dump, load")
            notebook.append(library)
            notebook.append("dataframe = pd.read_csv('"+filename+"')")
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
            notebook.append("dataframe_test = pd.read_csv('test.csv')")
            notebook.append("xs = dataframe_test["+str(x_cols)+"]")
            notebook.append("ys = dataframe_test['"+y+"']")
            notebook.append("predictions = model.predict(xs)")
            notebook.append("data_compare_test = pd.DataFrame({'Actual':ys, 'Predicted':predictions})")
            notebook.append("data_compare_test")
            sc.append(notebook) # actualiza el notebook

            '''
            Usando el modelo
            '''
            model = load("static/models/"+method+".joblib")
            dataframe_test = pd.read_csv("static/csv/droop_test.csv")
            xs = dataframe_test[['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8', 'y8', 'x9', 'y9', 'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15', 'y15', 'x16', 'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20', 'y20', 'x21', 'y21', 'x22', 'y22', 'x23', 'y23', 'x24', 'y24', 'x25', 'y25', 'x26', 'y26', 'x27', 'y27', 'x28', 'y28', 'x29', 'y29', 'x30', 'y30', 'x31', 'y31', 'x32', 'y32', 'x33', 'y33', 'x34', 'y34', 'x35', 'y35', 'x36', 'y36', 'x37', 'y37', 'x38', 'y38', 'x39', 'y39', 'x40', 'y40', 'x41', 'y41', 'x42', 'y42', 'x43', 'y43', 'x44', 'y44', 'x45', 'y45', 'x46', 'y46', 'x47', 'y47', 'x48', 'y48', 'x49', 'y49', 'x50', 'y50', 'x51', 'y51', 'x52', 'y52', 'x53', 'y53', 'x54', 'y54', 'x55', 'y55', 'x56', 'y56', 'x57', 'y57', 'x58', 'y58', 'x59', 'y59', 'x60', 'y60', 'x61', 'y61', 'x62', 'y62', 'x63', 'y63', 'x64', 'y64', 'x65', 'y65', 'x66', 'y66', 'x67', 'y67']]
            ys = dataframe_test['droop']
            predictions = model.predict(xs)
            data_compare_test = pd.DataFrame({"Actual":ys, "Predicted":predictions})
            print(data_compare_test)

            raise web.seeother('/classification_r')
        # except Exception as e:
        #     print(e.args)
        #     return render.error(e.args[0])

