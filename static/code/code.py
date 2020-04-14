# Libraries
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.mlab as mlab
from matplotlib.pyplot import figure, show
# Loading Dataframe 
dataframe = pd.read_csv('train.csv')
# Describe dataframe
dataframe.describe()
# Dataframe
dataframe
# Describe
dataframe.describe()
# Countplot
sn.countplot(data=dataframe, y='UCI')
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Countplot
sn.countplot(data=dataframe, y='ENTIDAD_NAC')
# Countplot
sn.countplot(data=dataframe, y='NACIONALIDAD')
# Describe
dataframe.describe()
# Describe
dataframe.describe()
# Describe
dataframe.describe()
# Countplot
sn.countplot(data=dataframe, y='FECHA_ACTUALIZACION')
# Countplot
sn.countplot(data=dataframe, y='FECHA_ACTUALIZACION')
# Countplot
sn.countplot(data=dataframe, y='FECHA_INGRESO')
# Countplot
sn.countplot(data=dataframe, y='FECHA_INGRESO')
# Histogram de EDAD
sn.distplot(dataframe[EDAD])
# Countplot
sn.countplot(data=dataframe, y='EDAD')
# Describe 'FECHA_ACTUALIZACION
dataframe['FECHA_ACTUALIZACION'].describe()
# Countplot
sn.countplot(data=dataframe, y='FECHA_ACTUALIZACION')
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Heatmap corr
correlation = dataframe.corr()
sn.heatmap(correlation,annot=True)
# Heatmap nulls
sn.heatmap(dataframe.isnull())
# Countplot
sn.countplot(data=dataframe, y='FECHA_DEF')
# Describe 'FECHA_DEF
dataframe['FECHA_DEF'].describe()
df = pd.read_csv('static/csv/train.csv')
df['def'] = df[['FECHA_DEF']].apply(lambda row: 'FECHA_DEF' != '9999-99-99'), axis=1)

df = pd.read_csv('static/csv/train.csv')
df['def'] = df[['FECHA_DEF']].apply(lambda row: 'FECHA_DEF' != '9999-99-99', axis=1)

# Histogram de def
sn.distplot(dataframe[def])
# Countplot
sn.countplot(data=dataframe, y='def')
# Describe 'def
dataframe['def'].describe()
# Drop
dataframe.drop(['def'],axis=1,inplace=True)
df = pd.read_csv('static/csv/train.csv')
def function(row):
    result = 0
    if row == '9999-99-99':
	result = 0
    else:
	result = 1
    return result

df['def'].apply(function,axis=1)
df = pd.read_csv('static/csv/train.csv')
def function(row):
    result = 0
    if row == '9999-99-99':
        result = 0
    else:
        result = 1
    return result

df['def'].apply(function,axis=1)
df = pd.read_csv('static/csv/train.csv')
def function(row):
    result = 0
    if row == '9999-99-99':
        result = 0
    else:
        result = 1
    return result

df['def']= function
# Countplot
sn.countplot(data=dataframe, y='def')
# Drop
dataframe.drop(['def'],axis=1,inplace=True)
df = pd.read_csv('static/csv/train.csv')
def function(row):
    result = 0
    if row == '9999-99-99':
        result = 0
    else:
        result = 1
    return result

df['def']= function()
df = pd.read_csv('static/csv/train.csv')
def function(row):
    result = 0
    if row == '9999-99-99':
        result = 0
    else:
        result = 1
    return result

df['def']= function(df['FECHA_DEF'])
df = pd.read_csv('static/csv/train.csv')
def function(row):
    result = 0
    if row == '9999-99-99':
        result = 0
    else:
        result = 1
    return result

df['def']= function(df['FECHA_DEF'],axis=1)
df = pd.read_csv('static/csv/train.csv')
def function(row):
    result = 0
    if row == '9999-99-99':
        result = 0
    else:
        result = 1
    return result

df['def']= function(df['FECHA_DEF'])
df = pd.read_csv('static/csv/train.csv')
def function(row):
    result = 0
    if row == '9999-99-99':
        result = 0
    else:
        result = 1
    return result

df['def']= df['FECHA_DEF'].apply(funcion,axis=1)
df = pd.read_csv('static/csv/train.csv')
def function(row):
    result = 0
    if row == '9999-99-99':
        result = 0
    else:
        result = 1
    return result

df['def']= df['FECHA_DEF'].apply(function,axis=1)
df = pd.read_csv('static/csv/train.csv')
def function(row):
    result = 0
    if row == '9999-99-99':
        result = 0
    else:
        result = 1
    return result

df['def']= df['FECHA_DEF'].apply(function)
# Countplot
sn.countplot(data=dataframe, y='def')
# Describe 'def
dataframe['def'].describe()
# Countplot
sn.countplot(data=dataframe, y='def')
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return = 0
    else:
        rerurn = 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO]].apply(function)
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        rerurn 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO]].apply(function)
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO]].apply(function)
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO']].apply(function)
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col'].apply(function(df['FECHA_DEF'],df['FECHA_INGRESO'])
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col'].apply(function(df['FECHA_DEF'],df['FECHA_INGRESO']))
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col'] = df.apply(function(df['FECHA_DEF'],df['FECHA_INGRESO']))
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col'] = df[['FECHA_DEF','FECHA_INGRESO]].apply(function(df['FECHA_DEF'],df['FECHA_INGRESO']))
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col'] = df[['FECHA_DEF','FECHA_INGRESO]].apply(function(var1=df['FECHA_DEF'],var2=df['FECHA_INGRESO']))
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col'] = function(var1=df['FECHA_DEF'],var2=df['FECHA_INGRESO'])
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col'] = function(df['FECHA_DEF'],df['FECHA_INGRESO'])
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO]].apply(function)
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO']].apply(function)
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO']].apply(function(df.FECHA_DEF,df.FECHA_INGRESO))
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO']].apply(function(df.FECHA_DEF,df.FECHA_INGRESO,axis=1))
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO']].apply(function(df.FECHA_DEF,df.FECHA_INGRESO),axis=1)
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    print(var1,var2)
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO']].apply(function(df.FECHA_DEF,df.FECHA_INGRESO),axis=1)
df = pd.read_csv('static/csv/train.csv')
def function(var1):
    if var1 == '9999-99-99':
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF']].apply(function)
df = pd.read_csv('static/csv/train.csv')
def function(var1):
    if var1 == '9999-99-99':
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF']].apply(function,axis=1)
df = pd.read_csv('static/csv/train.csv')
def function(var1):
    print(var1)
    if var1 == '9999-99-99':
        return 0
    else:
        return 1

df['new_col']= df['FECHA_DEF'].apply(function,axis=1)
df = pd.read_csv('static/csv/train.csv')
def function(var1):
    print(var1)
    if var1 == '9999-99-99':
        return 0
    else:
        return 1

df['new_col']= df['FECHA_DEF'].apply(function)
df = pd.read_csv('static/csv/train.csv')
def function(var1):
    if var1 == '9999-99-99':
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF']].apply(function,axis=1)
df = pd.read_csv('static/csv/train.csv')
def function(var1):
    if var1 == '9999-99-99':
        return 0
    else:
        return 1

df['new_col']= df['FECHA_DEF'].apply(function)
# Drop
dataframe.drop(['new_col'],axis=1,inplace=True)
# Drop
dataframe.drop(['def'],axis=1,inplace=True)
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    print(var1,var2)
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO']].apply(function(df[['FECHA_DEF','FECHA_INGRESO']]))
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    print(var1,var2)
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO']].apply(function(df['FECHA_DEF',df['FECHA_INGRESO']))
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    print(var1,var2)
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= df[['FECHA_DEF','FECHA_INGRESO']].apply(function(df['FECHA_DEF'],df['FECHA_INGRESO']))
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    print(var1,var2)
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col']= function(df['FECHA_DEF'],df['FECHA_INGRESO'])
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    print(var1,var2)
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col'] = df.apply(lambda x: function(x['FECHA_DEF'],x['FECHA_INGRESO']), axis=1)

df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    print(var1,var2)
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col'] = df.apply(lambda x: function(x['FECHA_DEF'],x['FECHA_INGRESO']), axis=1)
df = pd.read_csv('static/csv/train.csv')
def function(var1, var2):
    if var1 > var2:
        return 0
    else:
        return 1

df['new_col'] = df.apply(lambda x: function(x['FECHA_DEF'],x['FECHA_INGRESO']), axis=1)
# Describe 'new_col
dataframe['new_col'].describe()
# Drop
dataframe.drop(['new_col'],axis=1,inplace=True)
df = pd.read_csv('static/csv/train.csv')
def function(row):
   if row == 1:
       return "H"
   else:
       return "M"
df['sex_data'] = df.apply(lambda x: function(x['SEXO']), axis=1)

# Describe 'sex_data
dataframe['sex_data'].describe()
# Countplot
sn.countplot(data=dataframe, y='sex_data')
