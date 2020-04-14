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
# Histogram de ORIGEN
sn.distplot(dataframe[ORIGEN])
# Histogram de ENTIDAD_RES
sn.distplot(dataframe[ENTIDAD_RES])
# Histogram de ORIGEN
sn.distplot(dataframe[ORIGEN])
# Histogram de OTRO_CASO
sn.distplot(dataframe[OTRO_CASO])
# Histogram de RENAL_CRONICA
sn.distplot(dataframe[RENAL_CRONICA])
# Histogram de SEXO
sn.distplot(dataframe[SEXO])
# Histogram de MUNICIPIO_RES
sn.distplot(dataframe[MUNICIPIO_RES])
# Histogram de MIGRANTE
sn.distplot(dataframe[MIGRANTE])
# Histogram de PAIS_NACIONALIDAD
sn.distplot(dataframe[PAIS_NACIONALIDAD])
# Histogram de FECHA_ACTUALIZACION
sn.distplot(dataframe[FECHA_ACTUALIZACION])
# Histogram de PAIS_NACIONALIDAD
sn.distplot(dataframe[PAIS_NACIONALIDAD])
# Countplot
sn.countplot(data=dataframe, y='MUNICIPIO_RES')
# Histogram de ORIGEN
sn.distplot(dataframe[ORIGEN])
# Countplot
sn.countplot(data=dataframe, y='SECTOR')
# Histogram de SECTOR
sn.distplot(dataframe[SECTOR])
# Histogram de ENTIDAD_RES
sn.distplot(dataframe[ENTIDAD_RES])
# Countplot
sn.countplot(data=dataframe, y='MUNICIPIO_RES')
# Histogram de MUNICIPIO_RES
sn.distplot(dataframe[MUNICIPIO_RES])
# Histogram de ORIGEN
sn.distplot(dataframe[ORIGEN])
# Histogram de SEXO
sn.distplot(dataframe[SEXO])
# Histogram de FECHA_INGRESO
sn.distplot(dataframe[FECHA_INGRESO])
# Histogram de PAIS_ORIGEN
sn.distplot(dataframe[PAIS_ORIGEN])
# Histogram de ENTIDAD_NAC
sn.distplot(dataframe[ENTIDAD_NAC])
# Histogram de EDAD
sn.distplot(dataframe[EDAD])
# Heatmap nulls
sn.heatmap(dataframe.isnull())
