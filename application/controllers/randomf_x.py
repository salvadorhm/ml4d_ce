import web  # pip install web.py
import webdataminingtool
import csv  # CSV parser
import json  # json parser
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import matplotlib.mlab as mlab

from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing as pp

from matplotlib.pyplot import figure, show
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from application.controllers.save_code import SaveCode
sc = SaveCode()


render = web.template.render('application/views/', base="master")

class RandomfX:

    file = 'static/csv/temp.csv'  # define el archivo donde se almacenan los datos

    def __init__(self):  # Método inicial o constructor de la clase
        pass  # Simplemente continua con la ejecución

    def GET(self):
        try:
            dataframe = pd.read_csv(self.file)
            cols = list(dataframe)
            y = webdataminingtool.sessions['y']

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
            return render.randomf_x(columns,types,nulls,correlation)
        except Exception as e:
            print(e.args)
            return render.error(e.args[0])

    def POST(self):
        # try:
            try:
                filename = webdataminingtool.file['filename']
            except Exception as e:
                filename = "temp.csv"
            y = webdataminingtool.sessions['y']
            form = web.input(column = [''])
            x_cols = form.column
            webdataminingtool.sessions['x']=list(x_cols)
            dataframe = pd.read_csv(self.file)

            df_x = dataframe[x_cols]
            df_y = dataframe[y]

            x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)

            varianzas = x_train.describe().loc['std',:]
            medias = x_train.describe().loc['mean',:]
            
            model = RandomForestClassifier(n_estimators=80)
            # model = RandomForestClassifier(max_depth = 5, random_state = 101, criterion = 'gini', n_estimators = 50, min_samples_split = 5, min_samples_leaf = 2, max_features = 'log2')
            model.fit(x_train,y_train)
            predictions = model.predict(x_test)

            report = classification_report(y_test, predictions)
            confusion = confusion_matrix(y_test, predictions)

            importances = model.feature_importances_
            indices = np.argsort(importances)

            features = x_train.columns

            r = pd.DataFrame({"features":features[indices], "importance":importances[indices]})
            results = r.sort_values(by='importance',ascending=False)

            compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})

            plt.figure()
            width=20
            height=8
            figure(figsize=(width,height))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), features[indices])
            plt.xlabel('Relative Importance')
            image_name = "static/images/randomf.png"
            plt.savefig(image_name)

            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            sn.distplot(varianzas, bins = 200, hist = True, kde = True, color = 'g')
            image_name = "static/images/varianzas.png"
            plt.savefig(image_name)

            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            sn.distplot(medias, bins = 200, hist = True, kde = True)
            image_name = "static/images/medias.png"
            plt.savefig(image_name)

            #confusion_matrix
            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            conf_matrix = pd.DataFrame(data=confusion,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
            sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
            image_name = "static/images/confusion_matrix.png"
            plt.savefig(image_name)

            code = []
            code.append("import numpy as np")
            code.append("\n")
            code.append("from sklearn.metrics import classification_report, confusion_matrix")
            code.append("\n")
            code.append("from sklearn.model_selection import train_test_split")
            code.append("\n")
            code.append("from sklearn.ensemble import RandomForestClassifier")
            code.append("\n")
            code.append("dataframe = pd.read_csv("+filename+")")
            code.append("\n")
            code.append("df_x = dataframe["+str(x_cols)+"]")
            code.append("\n")
            code.append("df_y = dataframe['"+y+"']")
            code.append("\n")
            code.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            code.append("\n")
            code.append("model = RandomForestClassifier(n_estimators=80)")
            code.append("\n")
            code.append("model.fit(x_train,y_train)")
            code.append("\n")
            code.append("predictions = model.predict(x_test)")
            code.append("\n")
            code.append("classification_report(y_test, predictions)")
            code.append("\n")
            code.append("confusion_matrix(y_test, predictions)")
            code.append("\n")
            code.append("importances = model.feature_importances_")
            code.append("\n")
            code.append("indices = np.argsort(importances)")
            code.append("\n")
            code.append("features = x_train.columns")

            total_cols = len(x_cols)
            total_cols_20 = int(abs(total_cols * 0.20))

            webdataminingtool.randomf= {}
            webdataminingtool.randomf['filename']= filename
            webdataminingtool.randomf['y'] = y 
            webdataminingtool.randomf['x'] = list(x_cols)
            webdataminingtool.randomf['x_train.describe()['+str(x_cols[0])+"]"] = x_train.describe().to_dict()[x_cols[0]]
            webdataminingtool.randomf['N_estimators'] = 80
            webdataminingtool.randomf['Report'] = report
            webdataminingtool.randomf['Confusion matrix'] = list(confusion)
            webdataminingtool.randomf['Score'] = model.score(x_test,y_test)
            webdataminingtool.randomf['Accuracy score'] = accuracy_score(y_test, predictions)

            webdataminingtool.randomf['Real test values'] = list(compare.Actual.head(10))
            webdataminingtool.randomf['Predicted values'] = list(compare.Predicted.head(10))
            webdataminingtool.randomf['Python'] = ''.join(code)
            webdataminingtool.randomf['Features'] = list(results.features)[0:total_cols_20]
            webdataminingtool.randomf['Importance'] = list(results.importance)[0:total_cols_20]


            '''
            ----------------------------------------------------------------------------------------
            Normalizado
            ----------------------------------------------------------------------------------------
            '''
            # scaler_x = pp.StandardScaler(copy = True)
            # df_nor_x = scaler_x.fit_transform(df_x)

            df_nor_x = (df_x - df_x.mean())/df_x.std()
            # df_nor_x = pp.normalize(df_x)

            x_train, x_test, y_train, y_test = train_test_split(df_nor_x,df_y,test_size=0.3,random_state=42)
            
            varianzas = x_train.describe().loc['std',:]
            medias = x_train.describe().loc['mean',:]

            model = RandomForestClassifier(n_estimators=80)
            # model = RandomForestClassifier(max_depth = 5, random_state = 101, criterion = 'gini', n_estimators = 50, min_samples_split = 5, min_samples_leaf = 2, max_features = 'log2')
            
            model.fit(x_train,y_train)
            
            predictions = model.predict(x_test)

            report = classification_report(y_test, predictions)
            confusion = confusion_matrix(y_test, predictions)

            importances = model.feature_importances_
            indices = np.argsort(importances)

            features = x_train.columns


            r = pd.DataFrame({"features":features[indices], "importance":importances[indices]})
            results = r.sort_values(by='importance',ascending=False)

            compare = pd.DataFrame({"Actual":y_test, "Predicted":predictions})

            plt.figure()
            width=20
            height=8
            figure(figsize=(width,height))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='b', align='center')
            plt.yticks(range(len(indices)), features[indices])
            plt.xlabel('Relative Importance')
            image_name = "static/images/randomf_nor.png"
            plt.savefig(image_name)

            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            sn.distplot(varianzas, bins = 200, hist = True, kde = True, color = 'g')
            image_name = "static/images/varianzas_nor.png"
            plt.savefig(image_name)

            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            sn.distplot(medias, bins = 200, hist = True, kde = True)
            image_name = "static/images/medias_nor.png"
            plt.savefig(image_name)

            #confusion_matrix
            figure()
            width=20
            height=8
            figure(figsize=(width,height))
            conf_matrix=pd.DataFrame(data=confusion,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
            sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
            image_name = "static/images/confusion_matrix_nor.png"
            plt.savefig(image_name)

            code = []
            code.append("import numpy as np")
            code.append("\n")
            code.append("from sklearn.metrics import classification_report, confusion_matrix")
            code.append("\n")
            code.append("from sklearn.model_selection import train_test_split")
            code.append("\n")
            code.append("from sklearn.ensemble import RandomForestClassifier")
            code.append("\n")
            code.append("dataframe = pd.read_csv("+filename+")")
            code.append("\n")
            code.append("df_x = dataframe["+str(x_cols)+"]")
            code.append("\n")
            code.append("df_y = dataframe['"+y+"']")
            code.append("\n")
            code.append("df_nor_x = (df_x - df_x.mean())/df_x.std()")
            code.append("\n")
            code.append("x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)")
            code.append("\n")
            code.append("model = RandomForestClassifier(n_estimators=80)")
            code.append("\n")
            code.append("model.fit(x_train,y_train)")
            code.append("\n")
            code.append("predictions = model.predict(x_test)")
            code.append("\n")
            code.append("classification_report(y_test, predictions)")
            code.append("\n")
            code.append("confusion_matrix(y_test, predictions)")
            code.append("\n")
            code.append("importances = model.feature_importances_")
            code.append("\n")
            code.append("indices = np.argsort(importances)")
            code.append("\n")
            code.append("features = x_train.columns")

            total_cols = len(x_cols)
            total_cols_20 = int(abs(total_cols * 0.20))

            webdataminingtool.randomf_nor = {}
            webdataminingtool.randomf_nor['filename']= filename
            webdataminingtool.randomf_nor['y'] = y 
            webdataminingtool.randomf_nor['x'] = list(x_cols)
            # webdataminingtool.randomf_nor['x_train.describe()['+str(x_cols[0])+"]"] = pd.DataFrame(x_train).describe().to_dict()[x_cols[0]]
            webdataminingtool.randomf_nor['N_estimators'] = 80
            webdataminingtool.randomf_nor['Report'] = report
            webdataminingtool.randomf_nor['Confusion matrix'] = list(confusion)
            webdataminingtool.randomf_nor['Score'] = model.score(x_test,y_test)
            webdataminingtool.randomf_nor['Accuracy score'] = accuracy_score(y_test, predictions)

            webdataminingtool.randomf_nor['Real test values'] = list(compare.Actual.head(10))
            webdataminingtool.randomf_nor['Predicted values'] = list(compare.Predicted.head(10))
            webdataminingtool.randomf_nor['Python'] = ''.join(code)
            webdataminingtool.randomf_nor['Features max 20%'] = list(results.features)[0:total_cols_20]
            webdataminingtool.randomf_nor['Importance max 20%'] = list(results.importance)[0:total_cols_20]

            raise web.seeother('/randomf_r')
        # except Exception as e:
        #     print(e.args)
        #     return render.error(e.args[0])


  